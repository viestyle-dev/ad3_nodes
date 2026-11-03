"""Continuous (gap-free) oscilloscope capture from an Analog Discovery 3."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from ctypes import byref, c_byte, c_double, c_int

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc, replace_node_output_port
from nonoha_core.port import StaticTypeOutputPort

from . import dwf_binding as dwf
from .device_block import Ad3DeviceBlock

logger = logging.getLogger(__name__)

#: Per-channel scratch buffer for one FDwfAnalogInStatusData call. At 96 kHz a
#: 1 ms poll yields ~96 samples, so this is ~2700x headroom; anything beyond it
#: would already have been reported by the SDK as lost.
_CHUNK_CAPACITY = 262144

#: Seconds to let the input offset settle before arming, per the SDK samples.
_OFFSET_SETTLE_S = 2.0

#: Bounds for the adaptive poll interval. The reader sleeps roughly a quarter
#: pack between polls, so a pack is assembled in a handful of SDK reads instead
#: of thousands of sub-millisecond ones. Record mode buffers on the device
#: while the host sleeps (hundreds of ms of headroom at audio rates), so the
#: 10 ms ceiling costs no data — `lost` would report it if it ever did.
_POLL_SLEEP_MIN_S = 0.001
_POLL_SLEEP_MAX_S = 0.010


def make_factory_ad3_scope_node(device_block: Ad3DeviceBlock):
    """Build an Ad3Scope factory bound to the shared device block."""

    def factory(name: str):
        return Ad3ScopeNode(name, device_block)

    return factory


class Ad3ScopeNode(BaseNode):
    """Streams Analog Discovery scope channels into the graph without gaps.

    Uses the SDK's record acquisition mode, which is the only mode that streams
    continuously; ``FDwfAnalogInStatusRecord`` additionally reports how many
    samples the device had to drop, which is what separates "the device under
    test glitched" from "the measurement rig could not keep up".

    The scope inputs of an Analog Discovery are differential pairs, so a
    bridge-tied speaker output connects directly: hot to ``1+``, cold to ``1-``.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Continuously record Analog Discovery scope channels (record mode). "
                "Inputs are differential: C1 = 1+/1-, C2 = 2+/2-."
            ),
            category="AD3",
            inputs={},
            outputs={
                "series": "Captured samples in volts, one value per enabled channel.",
                "acq_status": (
                    "Cumulative sample accounting: lost (device could not keep up), "
                    "corrupted (device flagged the samples), "
                    "dropped (this node's queue overflowed)."
                ),
            },
        )

    PROPERTIES = {
        "device_index": {
            "description": "Device index to open. -1 selects the first available device.",
            "data_type": "i32",
            "default_value": -1,
        },
        "sample_rate": {
            "description": "Acquisition rate in Hz.",
            "data_type": "f64",
            "default_value": 96000.0,
        },
        "channels": {
            "description": "Number of scope channels to record (1 or 2).",
            "data_type": "i32",
            "default_value": 2,
        },
        "range_v": {
            "description": "Input range in volts peak-to-peak.",
            "data_type": "f64",
            "default_value": 5.0,
        },
        "pack_size": {
            "description": "Samples per dispatched block.",
            "data_type": "i32",
            "default_value": 1024,
        },
    }

    def __init__(self, name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, "Ad3Scope")

        self._device_block = device_block

        self._device_index = -1
        self._sample_rate = 96000.0
        self._channels = 2
        self._range_v = 5.0
        self._pack_size = 1024

        self._output_series = self._make_series_port(self._channels)
        self._output_acq_status = StaticTypeOutputPort(
            "acq_status",
            [PortValueType(DataType.Uint32)] * 3,
            value_name_list=["lost", "corrupted", "dropped"],
        )
        self.outputs[self._output_series.name] = self._output_series
        self.outputs[self._output_acq_status.name] = self._output_acq_status

        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._reader_thread: threading.Thread | None = None
        self._drain_task: asyncio.Task | None = None
        self._stop_event = threading.Event()
        self._hdwf: int | None = None

        # Cumulative counters (reader thread writes, drain task reads).
        self._lost = 0
        self._corrupted = 0
        self._dropped = 0

    @staticmethod
    def _make_series_port(channels: int) -> StaticTypeOutputPort:
        names = [f"ch{i + 1}" for i in range(channels)]
        return StaticTypeOutputPort(
            "series",
            [PortValueType(DataType.Float32)] * channels,
            value_name_list=names,
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def activate(self):
        self._main_loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stop_event.clear()
        self._drain_task = self._main_loop.create_task(self._drain_loop())
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"{self.name}_ad3_scope"
        )
        self._reader_thread.start()

    def finalize(self):
        self._stop_event.set()
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=5.0)
        self._reader_thread = None

    def set_properties(self, properties: dict[str, PropertyDataType]):
        channels = self._channels

        if "device_index" in properties:
            self._device_index = int(properties["device_index"])
        if "sample_rate" in properties:
            self._sample_rate = float(properties["sample_rate"])
        if "channels" in properties:
            channels = max(1, min(2, int(properties["channels"])))
        if "range_v" in properties:
            self._range_v = float(properties["range_v"])
        if "pack_size" in properties:
            self._pack_size = max(1, int(properties["pack_size"]))

        if channels != self._channels:
            new_port = self._make_series_port(channels)
            replace_node_output_port(self, self._output_series, new_port)
            self._output_series = new_port
            self._channels = channels

        self.properties["device_index"] = self._device_index
        self.properties["sample_rate"] = self._sample_rate
        self.properties["channels"] = self._channels
        self.properties["range_v"] = self._range_v
        self.properties["pack_size"] = self._pack_size

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

    # ── Reader thread ──────────────────────────────────────────────────────

    def _configure_acquisition(self, lib, hdwf: int) -> None:
        """Arm record-mode acquisition. Caller must hold the device lock."""
        for ch in range(self._channels):
            lib.FDwfAnalogInChannelEnableSet(hdwf, c_int(ch), c_int(1))
            lib.FDwfAnalogInChannelRangeSet(hdwf, c_int(ch), dwf.as_double(self._range_v))
            lib.FDwfAnalogInChannelFilterSet(hdwf, c_int(ch), c_int(dwf.FILTER_DECIMATE))
        # Disable the channels we are not using so they do not consume bandwidth.
        for ch in range(self._channels, 2):
            lib.FDwfAnalogInChannelEnableSet(hdwf, c_int(ch), c_int(0))

        lib.FDwfAnalogInAcquisitionModeSet(hdwf, c_int(dwf.ACQMODE_RECORD))
        lib.FDwfAnalogInFrequencySet(hdwf, dwf.as_double(self._sample_rate))
        # Length 0 records indefinitely — the graph decides when to stop.
        lib.FDwfAnalogInRecordLengthSet(hdwf, dwf.as_double(0.0))
        lib.FDwfAnalogInConfigure(hdwf, c_int(1), c_int(0))

    def _reader_loop(self):
        try:
            hdwf = self._device_block.acquire(self._device_index)
        except Exception as exc:
            logger.error("%s: cannot open Analog Discovery: %s", self.name, exc)
            return

        self._hdwf = hdwf
        lib = self._device_block.lib
        device_index = self._device_index
        channels = self._channels
        pack_size = self._pack_size
        poll_sleep = min(
            max(pack_size / self._sample_rate / 4.0, _POLL_SLEEP_MIN_S),
            _POLL_SLEEP_MAX_S,
        )

        buffers = [(c_double * _CHUNK_CAPACITY)() for _ in range(channels)]
        pending: list[list[np.ndarray]] = [[] for _ in range(channels)]
        pending_count = 0

        sts = c_byte()
        c_available = c_int()
        c_lost = c_int()
        c_corrupted = c_int()
        started = False

        try:
            with self._device_block.lock:
                self._configure_acquisition(lib, hdwf)

            # The input offset needs to settle before the first sample is valid.
            if self._stop_event.wait(_OFFSET_SETTLE_S):
                return

            with self._device_block.lock:
                lib.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(1))
            logger.info(
                "%s: recording %d ch @ %.0f Hz (range %.2f V)",
                self.name,
                channels,
                self._sample_rate,
                self._range_v,
            )

            while not self._stop_event.is_set():
                with self._device_block.lock:
                    lib.FDwfAnalogInStatus(hdwf, c_int(1), byref(sts))

                    if not started and sts.value in dwf.DWF_STATES_NOT_STARTED:
                        time.sleep(poll_sleep)
                        continue

                    lib.FDwfAnalogInStatusRecord(
                        hdwf, byref(c_available), byref(c_lost), byref(c_corrupted)
                    )
                    available = c_available.value
                    lost = c_lost.value
                    corrupted = c_corrupted.value

                    if available > _CHUNK_CAPACITY:
                        # Cannot happen at audio rates, but never read past the buffer.
                        lost += available - _CHUNK_CAPACITY
                        available = _CHUNK_CAPACITY

                    if available > 0:
                        # Every channel must be read against the same `available`
                        # count before the next FDwfAnalogInStatus call.
                        for ch in range(channels):
                            lib.FDwfAnalogInStatusData(
                                hdwf, c_int(ch), byref(buffers[ch]), c_int(available)
                            )

                started = True
                self._lost += lost
                self._corrupted += corrupted

                if available == 0:
                    time.sleep(poll_sleep)
                    continue

                for ch in range(channels):
                    block = np.frombuffer(
                        buffers[ch], dtype=np.float64, count=available
                    ).astype(np.float32)
                    pending[ch].append(block)
                pending_count += available

                while pending_count >= pack_size:
                    packed = [np.concatenate(p) for p in pending]
                    for ch in range(channels):
                        pending[ch] = [packed[ch][pack_size:]]
                    pending_count -= pack_size
                    self._enqueue([p[:pack_size] for p in packed])

                # Below a full pack there is nothing to dispatch yet, so let
                # samples accumulate on the device instead of spinning on the
                # SDK — record mode keeps them and reports any overflow as lost.
                if pending_count < pack_size:
                    time.sleep(poll_sleep)

        except Exception as exc:
            logger.error("%s: acquisition error: %s", self.name, exc)
        finally:
            try:
                with self._device_block.lock:
                    lib.FDwfAnalogInConfigure(hdwf, c_int(0), c_int(0))
            except Exception:
                pass
            self._hdwf = None
            self._device_block.release(device_index)

    def _enqueue(self, block: list[np.ndarray]) -> None:
        """Hand a packed block to the event loop (called from the reader thread)."""
        loop = self._main_loop
        queue = self._queue
        if loop is None or queue is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, block)
        except RuntimeError:
            # Loop already closed during shutdown.
            self._dropped += len(block[0]) if block else 0

    # ── Event loop ─────────────────────────────────────────────────────────

    async def _drain_loop(self):
        queue = self._queue
        assert queue is not None
        last_status: tuple[int, int, int] | None = None
        last_status_ts = 0.0
        try:
            while True:
                block = await queue.get()
                self._output_series.dispatch(block)
                # The counters are cumulative and almost always all zero, so
                # re-dispatching them with every block only pads the WebSocket
                # stream. Dispatch on change, plus a ~1 s heartbeat so a viewer
                # that connects mid-run still sees the current counts.
                status = (self._lost, self._corrupted, self._dropped)
                now = time.monotonic()
                if status != last_status or now - last_status_ts >= 1.0:
                    last_status = status
                    last_status_ts = now
                    self._output_acq_status.dispatch(
                        [np.array([v], dtype=np.uint32) for v in status]
                    )
                if self.parent_graph is not None:
                    self.parent_graph.flush_outputs()
        except asyncio.CancelledError:
            pass
        except Exception as exc:  # pragma: no cover
            logger.error("%s: dispatch error: %s", self.name, exc)
