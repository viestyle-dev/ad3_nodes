"""Analog Discovery 3 supply, digital I/O and UART as bench-control nodes.

These let a flow run a firmware bring-up rig without the WaveForms GUI, which
would otherwise hold the device and lock the measurement flows out:

- ``Ad3Supply``     positive supply V+ (e.g. 1.8 V for a sensor board)
- ``Ad3OpenDrain``  one DIO pin as an open-drain output (reset, BOOTSEL strap)
- ``Ad3UartRx``     UART receive on one DIO pin, line by line, optional log file

All three share the device handle through :class:`Ad3DeviceBlock` with the
scope and the generator.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from ctypes import byref, c_double, c_int, c_uint, create_string_buffer, sizeof
from pathlib import Path

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

from . import dwf_binding as dwf
from .device_block import Ad3DeviceBlock

logger = logging.getLogger(__name__)

#: UART poll period. 115200 baud is ~11.5 kB/s and the SDK buffers on-device,
#: so 10 ms polls read ~115 bytes each with ample headroom.
_UART_POLL_S = 0.010
_UART_BUFFER_BYTES = 8192


def make_factory_ad3_supply_node(device_block: Ad3DeviceBlock):
    """Build an Ad3Supply factory bound to the shared device block."""

    def factory(name: str):
        return Ad3SupplyNode(name, device_block)

    return factory


def make_factory_ad3_open_drain_node(device_block: Ad3DeviceBlock):
    """Build an Ad3OpenDrain factory bound to the shared device block."""

    def factory(name: str):
        return Ad3OpenDrainNode(name, device_block)

    return factory


def make_factory_ad3_uart_rx_node(device_block: Ad3DeviceBlock):
    """Build an Ad3UartRx factory bound to the shared device block."""

    def factory(name: str):
        return Ad3UartRxNode(name, device_block)

    return factory


class _Ad3Node(BaseNode):
    """Device acquire/release and the ``state`` output shared by the I/O nodes."""

    def __init__(self, name: str, type_name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, type_name)
        self._device_block = device_block
        self._device_index = -1
        self._hdwf: int | None = None

        self._output_state = StaticTypeOutputPort("state", [PortValueType(DataType.String)])
        self.outputs[self._output_state.name] = self._output_state

    def _acquire(self) -> bool:
        try:
            self._hdwf = self._device_block.acquire(self._device_index)
        except Exception as exc:
            logger.error("%s: cannot open Analog Discovery: %s", self.name, exc)
            self._dispatch_state("error")
            return False
        return True

    def _release(self) -> None:
        if self._hdwf is not None:
            self._device_block.release(self._device_index)
            self._hdwf = None

    def _dispatch_state(self, state: str) -> None:
        self._output_state.dispatch([np.array([state])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()


# ── Ad3Supply ──────────────────────────────────────────────────────────────


class Ad3SupplyNode(_Ad3Node):
    """Drives the Analog Discovery positive supply (V+) at a set voltage.

    The negative supply is kept off. The supply keeps running when the graph
    stops (the device block closes handles with "continue"), so stopping a
    flow never drops a rail under a powered target; set ``off_on_finalize``
    to switch it off instead.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Analog Discovery positive supply V+. Sets the voltage and switches the "
                "rail; the negative supply stays off."
            ),
            category="AD3",
            inputs={
                "enabled": "Bool. True switches V+ on at voltage_v, False switches it off.",
            },
            outputs={
                "state": "Supply state: off / on <measured> V / error.",
            },
        )

    PROPERTIES = {
        "device_index": {
            "description": "Device index to open. -1 selects the first available device.",
            "data_type": "i32",
            "default_value": -1,
        },
        "voltage_v": {
            "description": "V+ output voltage in volts (AD3: 0.5 to 5.0).",
            "data_type": "f64",
            "default_value": 1.8,
        },
        "start_on_activate": {
            "description": "Switch V+ on as soon as the graph runs, without waiting for the enabled port.",
            "data_type": "bool",
            "default_value": False,
        },
        "off_on_finalize": {
            "description": "Switch V+ off when the graph stops. Off by default so a stopped flow "
            "never removes a rail from a target that is still powered elsewhere.",
            "data_type": "bool",
            "default_value": False,
        },
    }

    def __init__(self, name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, "Ad3Supply", device_block)
        self._voltage_v = 1.8
        self._start_on_activate = False
        self._off_on_finalize = False
        self._on = False

        self._input_enabled = StaticTypeInputPort("enabled", [PortValueType(DataType.Bool)])
        self._input_enabled.subscribe(id(self), self._on_enabled)
        self.inputs[self._input_enabled.name] = self._input_enabled

    def activate(self):
        if not self._acquire():
            return
        self._apply(self._start_on_activate)

    def finalize(self):
        if self._hdwf is not None and self._off_on_finalize:
            try:
                self._apply(False)
            except Exception:
                pass
        self._release()

    def _clamped_voltage(self) -> float:
        v = min(max(self._voltage_v, dwf.VPLUS_MIN_V), dwf.VPLUS_MAX_V)
        if v != self._voltage_v:
            logger.warning("%s: voltage_v %.3f V clamped to %.3f V", self.name, self._voltage_v, v)
        return v

    def _apply(self, on: bool) -> None:
        if self._hdwf is None:
            return
        lib = self._device_block.lib
        hdwf = self._hdwf
        volts = self._clamped_voltage()
        measured = c_double()
        try:
            with self._device_block.lock:
                vp, vm = c_int(dwf.ANALOG_IO_CH_VPLUS), c_int(dwf.ANALOG_IO_CH_VMINUS)
                node_en, node_v = c_int(dwf.ANALOG_IO_NODE_ENABLE), c_int(dwf.ANALOG_IO_NODE_VOLTAGE)
                dwf.check(lib, lib.FDwfAnalogIOChannelNodeSet(hdwf, vp, node_v, dwf.as_double(volts)), "V+ voltage")
                dwf.check(lib, lib.FDwfAnalogIOChannelNodeSet(hdwf, vp, node_en, dwf.as_double(1.0 if on else 0.0)), "V+ enable")
                dwf.check(lib, lib.FDwfAnalogIOChannelNodeSet(hdwf, vm, node_en, dwf.as_double(0.0)), "V- disable")
                dwf.check(lib, lib.FDwfAnalogIOEnableSet(hdwf, c_int(1 if on else 0)), "supply master enable")
                dwf.check(lib, lib.FDwfAnalogIOConfigure(hdwf), "supply configure")
                if on:
                    time.sleep(0.05)  # let the rail settle before reading it back
                lib.FDwfAnalogIOStatus(hdwf)
                lib.FDwfAnalogIOChannelNodeStatus(hdwf, vp, node_v, byref(measured))
        except Exception as exc:
            logger.error("%s: supply configuration failed: %s", self.name, exc)
            self._dispatch_state("error")
            return

        self._on = on
        if on:
            logger.info("%s: V+ on, set %.3f V, measured %.3f V", self.name, volts, measured.value)
            self._dispatch_state(f"on {measured.value:.3f} V")
        else:
            logger.info("%s: V+ off", self.name)
            self._dispatch_state("off")

    def _on_enabled(self, values: list[PortValue]):
        self._apply(bool(values[0][0]))

    def set_properties(self, properties: dict[str, PropertyDataType]):
        previous_device_index = self._device_index
        if "device_index" in properties:
            self._device_index = int(properties["device_index"])
        if "voltage_v" in properties:
            self._voltage_v = float(properties["voltage_v"])
        if "start_on_activate" in properties:
            self._start_on_activate = bool(properties["start_on_activate"])
        if "off_on_finalize" in properties:
            self._off_on_finalize = bool(properties["off_on_finalize"])

        self.properties["device_index"] = self._device_index
        self.properties["voltage_v"] = self._voltage_v
        self.properties["start_on_activate"] = self._start_on_activate
        self.properties["off_on_finalize"] = self._off_on_finalize

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        if self.is_active:
            if self._device_index != previous_device_index and self._hdwf is not None:
                self._device_block.release(previous_device_index)
                self._hdwf = None
            if self._hdwf is None and not self._acquire():
                return
            if self._on:
                self._apply(True)  # pick up a new voltage while running


# ── Ad3OpenDrain ───────────────────────────────────────────────────────────


class Ad3OpenDrainNode(_Ad3Node):
    """One DIO pin used as an open-drain output: driven low, or released.

    The pin never drives high, so it can sit on a line the target also drives
    or pulls up (an MCU reset pin, a boot-mode strap shared with a flash chip
    select). Put ~1 kOhm in series anyway, against a wrong pin index.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Analog Discovery DIO as an open-drain output: low or high-impedance, never "
                "driven high. For target reset and boot-mode straps."
            ),
            category="AD3",
            inputs={
                "low": "Bool level. True holds the pin low, False releases it (high-impedance).",
                "pulse": "Bool. True pulls the pin low for pulse_ms, then releases it.",
            },
            outputs={
                "state": "Pin state: released / low / error.",
            },
        )

    PROPERTIES = {
        "device_index": {
            "description": "Device index to open. -1 selects the first available device.",
            "data_type": "i32",
            "default_value": -1,
        },
        "dio": {
            "description": "DIO pin number (AD3: 0 to 15).",
            "data_type": "i32",
            "default_value": 0,
        },
        "pulse_ms": {
            "description": "Low time for the pulse input, in milliseconds.",
            "data_type": "f64",
            "default_value": 100.0,
        },
    }

    def __init__(self, name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, "Ad3OpenDrain", device_block)
        self._dio = 0
        self._pulse_ms = 100.0
        self._held = False
        self._pulse_handle: asyncio.TimerHandle | None = None

        self._input_low = StaticTypeInputPort("low", [PortValueType(DataType.Bool)])
        self._input_low.subscribe(id(self), self._on_low)
        self.inputs[self._input_low.name] = self._input_low

        self._input_pulse = StaticTypeInputPort("pulse", [PortValueType(DataType.Bool)])
        self._input_pulse.subscribe(id(self), self._on_pulse)
        self.inputs[self._input_pulse.name] = self._input_pulse

    def activate(self):
        if not self._acquire():
            return
        self._drive(False)

    def finalize(self):
        if self._pulse_handle is not None:
            self._pulse_handle.cancel()
            self._pulse_handle = None
        if self._hdwf is not None:
            try:
                self._drive(False)  # never leave a reset asserted
            except Exception:
                pass
        self._release()

    def _drive(self, low: bool) -> None:
        if self._hdwf is None:
            return
        lib = self._device_block.lib
        hdwf = self._hdwf
        bit = 1 << self._dio
        enable = c_uint()
        output = c_uint()
        try:
            with self._device_block.lock:
                lib.FDwfDigitalIOOutputEnableGet(hdwf, byref(enable))
                lib.FDwfDigitalIOOutputGet(hdwf, byref(output))
                # Output latch is always 0 on this pin; only the enable toggles,
                # so the pin is either driven low or floating.
                dwf.check(lib, lib.FDwfDigitalIOOutputSet(hdwf, c_uint(output.value & ~bit)), "DIO output")
                new_enable = (enable.value | bit) if low else (enable.value & ~bit)
                dwf.check(lib, lib.FDwfDigitalIOOutputEnableSet(hdwf, c_uint(new_enable)), "DIO enable")
                dwf.check(lib, lib.FDwfDigitalIOConfigure(hdwf), "DIO configure")
        except Exception as exc:
            logger.error("%s: DIO%d update failed: %s", self.name, self._dio, exc)
            self._dispatch_state("error")
            return
        self._dispatch_state("low" if low else "released")

    def _on_low(self, values: list[PortValue]):
        self._held = bool(values[0][0])
        self._drive(self._held)

    def _on_pulse(self, values: list[PortValue]):
        if not bool(values[0][0]):
            return
        self._drive(True)
        if self._pulse_handle is not None:
            self._pulse_handle.cancel()
        loop = asyncio.get_running_loop()
        self._pulse_handle = loop.call_later(self._pulse_ms / 1000.0, self._end_pulse)
        logger.info("%s: DIO%d low for %.0f ms", self.name, self._dio, self._pulse_ms)

    def _end_pulse(self) -> None:
        self._pulse_handle = None
        if not self._held:
            self._drive(False)

    def set_properties(self, properties: dict[str, PropertyDataType]):
        previous_device_index = self._device_index
        previous_dio = self._dio
        if "device_index" in properties:
            self._device_index = int(properties["device_index"])
        if "dio" in properties:
            self._dio = min(max(int(properties["dio"]), 0), 15)
        if "pulse_ms" in properties:
            self._pulse_ms = max(float(properties["pulse_ms"]), 1.0)

        self.properties["device_index"] = self._device_index
        self.properties["dio"] = self._dio
        self.properties["pulse_ms"] = self._pulse_ms

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        if self.is_active:
            if self._hdwf is not None and (
                self._dio != previous_dio or self._device_index != previous_device_index
            ):
                # Release the old pin before moving to the new one.
                new_dio = self._dio
                self._dio = previous_dio
                self._drive(False)
                self._dio = new_dio
            if self._device_index != previous_device_index and self._hdwf is not None:
                self._device_block.release(previous_device_index)
                self._hdwf = None
            if self._hdwf is None and not self._acquire():
                return
            self._drive(self._held)


# ── Ad3UartRx ──────────────────────────────────────────────────────────────


class Ad3UartRxNode(_Ad3Node):
    """Receives a UART on one DIO pin and emits it line by line.

    Lines are also appended to ``log_file`` (with a host timestamp) when it is
    set, so a target's boot log can be read back after the fact by anything
    that can open a file.
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Analog Discovery UART receiver (8N1) on one DIO pin. Emits each received "
                "line; optionally appends them to a log file."
            ),
            category="AD3",
            outputs={
                "line": "One received text line, without the line ending.",
                "state": "Receiver state: receiving / error.",
            },
        )

    PROPERTIES = {
        "device_index": {
            "description": "Device index to open. -1 selects the first available device.",
            "data_type": "i32",
            "default_value": -1,
        },
        "rx_dio": {
            "description": "DIO pin wired to the target's TX (AD3: 0 to 15).",
            "data_type": "i32",
            "default_value": 1,
        },
        "baud": {
            "description": "Baud rate (8 data bits, no parity, 1 stop bit).",
            "data_type": "f64",
            "default_value": 115200.0,
        },
        "log_file": {
            "description": "Append received lines here with a timestamp. A relative path is placed "
            "in the graph's data folder. Empty disables the file.",
            "data_type": "str",
            "default_value": "",
        },
    }

    def __init__(self, name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, "Ad3UartRx", device_block)
        self._rx_dio = 1
        self._baud = 115200.0
        self._log_file = ""

        self._output_line = StaticTypeOutputPort("line", [PortValueType(DataType.String)])
        self.outputs[self._output_line.name] = self._output_line

        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue | None = None
        self._reader_thread: threading.Thread | None = None
        self._drain_task: asyncio.Task | None = None
        self._stop_event = threading.Event()

    def activate(self):
        if not self._acquire():
            return
        self._start()

    def finalize(self):
        self._stop()
        self._release()

    # ── Reader ─────────────────────────────────────────────────────────────

    def _start(self) -> None:
        self._main_loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        self._stop_event.clear()
        self._drain_task = self._main_loop.create_task(self._drain_loop())
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True, name=f"{self.name}_ad3_uart"
        )
        self._reader_thread.start()

    def _stop(self) -> None:
        self._stop_event.set()
        if self._drain_task is not None:
            self._drain_task.cancel()
            self._drain_task = None
        if self._reader_thread is not None and self._reader_thread.is_alive():
            self._reader_thread.join(timeout=2.0)
        self._reader_thread = None

    def _configure(self, lib, hdwf: int) -> None:
        count = c_int()
        parity = c_int()
        with self._device_block.lock:
            dwf.check(lib, lib.FDwfDigitalUartRateSet(hdwf, dwf.as_double(self._baud)), "UART rate")
            dwf.check(lib, lib.FDwfDigitalUartRxSet(hdwf, c_int(self._rx_dio)), "UART RX pin")
            dwf.check(lib, lib.FDwfDigitalUartBitsSet(hdwf, c_int(8)), "UART bits")
            dwf.check(lib, lib.FDwfDigitalUartParitySet(hdwf, c_int(0)), "UART parity")
            dwf.check(lib, lib.FDwfDigitalUartStopSet(hdwf, dwf.as_double(1.0)), "UART stop")
            # A zero-length read arms the receiver.
            dwf.check(lib, lib.FDwfDigitalUartRx(hdwf, None, c_int(0), byref(count), byref(parity)), "UART start")

    def _reader_loop(self) -> None:
        lib = self._device_block.lib
        hdwf = self._hdwf
        if hdwf is None:
            return
        try:
            self._configure(lib, hdwf)
        except Exception as exc:
            logger.error("%s: UART setup failed: %s", self.name, exc)
            self._enqueue(("state", "error"))
            return
        logger.info("%s: receiving on DIO%d at %.0f baud", self.name, self._rx_dio, self._baud)
        self._enqueue(("state", "receiving"))

        buf = create_string_buffer(_UART_BUFFER_BYTES + 1)
        count = c_int()
        parity = c_int()
        pending = ""
        log_path = self._resolve_log_path()
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("%s: logging to %s", self.name, log_path)

        while not self._stop_event.is_set():
            try:
                with self._device_block.lock:
                    ok = lib.FDwfDigitalUartRx(hdwf, buf, c_int(sizeof(buf) - 1), byref(count), byref(parity))
            except Exception as exc:  # pragma: no cover
                logger.error("%s: UART read raised: %s", self.name, exc)
                ok = 0
            if not ok:
                logger.error("%s: UART read failed: %s", self.name, dwf.last_error(lib))
                self._enqueue(("state", "error"))
                return
            if count.value > 0:
                pending += buf.raw[: count.value].decode("ascii", errors="replace")
                *lines, pending = pending.replace("\r\n", "\n").split("\n")
                for line in lines:
                    if log_path is not None:
                        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(f"{stamp} {line}\n")
                    self._enqueue(("line", line))
            if parity.value != 0:
                logger.warning("%s: UART framing/parity error %d", self.name, parity.value)
            self._stop_event.wait(_UART_POLL_S)

    def _resolve_log_path(self) -> Path | None:
        """Relative paths land in the graph's data folder, like report outputs."""
        if not self._log_file:
            return None
        path = Path(self._log_file)
        if path.is_absolute():
            return path
        parent_graph = self.parent_graph
        persist = getattr(parent_graph, "persist", None) if parent_graph is not None else None
        if persist is None:
            return path
        return persist.data_dir_path / path

    def _enqueue(self, item: tuple[str, str]) -> None:
        loop = self._main_loop
        queue = self._queue
        if loop is None or queue is None:
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass  # loop closed during shutdown

    async def _drain_loop(self):
        queue = self._queue
        assert queue is not None
        try:
            while True:
                kind, text = await queue.get()
                if kind == "line":
                    self._output_line.dispatch([np.array([text])])
                    if self.parent_graph is not None:
                        self.parent_graph.flush_outputs()
                else:
                    self._dispatch_state(text)
        except asyncio.CancelledError:
            pass

    def set_properties(self, properties: dict[str, PropertyDataType]):
        previous = (self._device_index, self._rx_dio, self._baud, self._log_file)
        previous_device_index = self._device_index
        if "device_index" in properties:
            self._device_index = int(properties["device_index"])
        if "rx_dio" in properties:
            self._rx_dio = min(max(int(properties["rx_dio"]), 0), 15)
        if "baud" in properties:
            self._baud = float(properties["baud"])
        if "log_file" in properties:
            self._log_file = str(properties["log_file"])

        self.properties["device_index"] = self._device_index
        self.properties["rx_dio"] = self._rx_dio
        self.properties["baud"] = self._baud
        self.properties["log_file"] = self._log_file

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        if self.is_active and (self._device_index, self._rx_dio, self._baud, self._log_file) != previous:
            self._stop()
            if self._device_index != previous_device_index and self._hdwf is not None:
                self._device_block.release(previous_device_index)
                self._hdwf = None
            if self._hdwf is None and not self._acquire():
                return
            self._start()
