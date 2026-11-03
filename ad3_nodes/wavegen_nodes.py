"""Analog Discovery 3 waveform generator (W1/W2) as a reference signal source."""

from __future__ import annotations

import logging
from ctypes import c_int

import numpy as np

from nonoha_core.data_type import DataType, PortValueType, PropertyDataType
from nonoha_core.node import BaseNode, NodeDoc
from nonoha_core.port import PortValue, StaticTypeInputPort, StaticTypeOutputPort

from . import dwf_binding as dwf
from .device_block import Ad3DeviceBlock

logger = logging.getLogger(__name__)


def make_factory_ad3_wavegen_node(device_block: Ad3DeviceBlock):
    """Build an Ad3WaveGen factory bound to the shared device block."""

    def factory(name: str):
        return Ad3WaveGenNode(name, device_block)

    return factory


class Ad3WaveGenNode(BaseNode):
    """Drives the Analog Discovery waveform generator as a gap-free reference.

    The waveform runs entirely on the device's own synthesizer — nothing is
    streamed from the host — so the injected signal is continuous by
    construction. That is what makes it usable as a reference for end-to-end
    capture tests: any discontinuity observed downstream came from the device
    under test, not from this generator.

    W1 is single-ended (referenced to GND). Set ``differential`` to drive W2
    with the same waveform 180 degrees out of phase, giving a true differential
    pair for a balanced microphone input (hot = W1, cold = W2).
    """

    @classmethod
    def describe(cls, properties=None):
        return NodeDoc(
            description=(
                "Analog Discovery waveform generator. Runs on-device, so the output "
                "is continuous regardless of host load. Optionally drives W2 in "
                "anti-phase for differential injection."
            ),
            category="AD3",
            inputs={
                "enabled": "Bool. True starts the generator, False returns it to the offset level.",
            },
            outputs={
                "state": "Generator state: idle/running/error.",
            },
        )

    PROPERTIES = {
        "device_index": {
            "description": "Device index to open. -1 selects the first available device.",
            "data_type": "i32",
            "default_value": -1,
        },
        "channel": {
            "description": "Primary output channel (0 = W1, 1 = W2).",
            "data_type": "i32",
            "default_value": 0,
        },
        "function": {
            "description": "Waveform shape: dc, sine, square, triangle, ramp_up, ramp_down, noise.",
            "data_type": "str",
            "default_value": "sine",
        },
        "frequency": {
            "description": "Waveform frequency in Hz.",
            "data_type": "f64",
            "default_value": 1000.0,
        },
        "amplitude_v": {
            "description": "Waveform amplitude in volts (peak). Mic inputs usually need an external divider.",
            "data_type": "f64",
            "default_value": 0.01,
        },
        "offset_v": {
            "description": "DC offset in volts.",
            "data_type": "f64",
            "default_value": 0.0,
        },
        "differential": {
            "description": "Also drive the other channel 180 degrees out of phase (balanced output).",
            "data_type": "bool",
            "default_value": False,
        },
        "start_on_activate": {
            "description": "Start generating as soon as the graph runs, without waiting for the enabled port.",
            "data_type": "bool",
            "default_value": True,
        },
    }

    def __init__(self, name: str, device_block: Ad3DeviceBlock):
        super().__init__(name, "Ad3WaveGen")

        self._device_block = device_block

        self._device_index = -1
        self._channel = 0
        self._function = "sine"
        self._frequency = 1000.0
        self._amplitude_v = 0.01
        self._offset_v = 0.0
        self._differential = False
        self._start_on_activate = True

        self._input_enabled = StaticTypeInputPort("enabled", [PortValueType(DataType.Bool)])
        self._input_enabled.subscribe(id(self), self._on_enabled)
        self.inputs[self._input_enabled.name] = self._input_enabled

        self._output_state = StaticTypeOutputPort("state", [PortValueType(DataType.String)])
        self.outputs[self._output_state.name] = self._output_state

        self._hdwf: int | None = None
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def activate(self):
        try:
            self._hdwf = self._device_block.acquire(self._device_index)
        except Exception as exc:
            logger.error("%s: cannot open Analog Discovery: %s", self.name, exc)
            self._dispatch_state("error")
            return
        self._apply(start=self._start_on_activate)

    def finalize(self):
        if self._hdwf is not None:
            try:
                self._stop()
            except Exception:
                pass
            self._device_block.release(self._device_index)
            self._hdwf = None

    # ── Channel helpers ────────────────────────────────────────────────────

    def _channels(self) -> list[tuple[int, float]]:
        """Return (channel index, phase in degrees) pairs to drive."""
        primary = self._channel
        if not self._differential:
            return [(primary, 0.0)]
        return [(primary, 0.0), (1 - primary, 180.0)]

    def _apply(self, start: bool) -> None:
        """Push the current properties to the hardware."""
        if self._hdwf is None:
            return
        lib = self._device_block.lib
        hdwf = self._hdwf
        func = dwf.FUNCTIONS.get(self._function)
        if func is None:
            logger.error(
                "%s: unknown function %r (expected one of %s)",
                self.name,
                self._function,
                ", ".join(sorted(dwf.FUNCTIONS)),
            )
            self._dispatch_state("error")
            return

        node = c_int(dwf.ANALOG_OUT_NODE_CARRIER)
        try:
            with self._device_block.lock:
                for ch, phase in self._channels():
                    ch_c = c_int(ch)
                    lib.FDwfAnalogOutNodeEnableSet(hdwf, ch_c, node, c_int(1))
                    lib.FDwfAnalogOutNodeFunctionSet(hdwf, ch_c, node, c_int(func))
                    lib.FDwfAnalogOutNodeFrequencySet(
                        hdwf, ch_c, node, dwf.as_double(self._frequency)
                    )
                    lib.FDwfAnalogOutNodeAmplitudeSet(
                        hdwf, ch_c, node, dwf.as_double(self._amplitude_v)
                    )
                    lib.FDwfAnalogOutNodeOffsetSet(
                        hdwf, ch_c, node, dwf.as_double(self._offset_v)
                    )
                    lib.FDwfAnalogOutNodePhaseSet(hdwf, ch_c, node, dwf.as_double(phase))
                    # Rest at the offset level rather than going high-impedance,
                    # so a stopped generator presents a defined DC level.
                    lib.FDwfAnalogOutIdleSet(
                        hdwf, ch_c, c_int(dwf.DWF_ANALOG_OUT_IDLE_OFFSET)
                    )
                    lib.FDwfAnalogOutConfigure(hdwf, ch_c, c_int(1 if start else 0))
        except Exception as exc:
            logger.error("%s: configuration failed: %s", self.name, exc)
            self._dispatch_state("error")
            return

        self._running = start
        if start:
            logger.info(
                "%s: generating %s %.3f Hz %.4f Vp on %s",
                self.name,
                self._function,
                self._frequency,
                self._amplitude_v,
                ", ".join(f"W{ch + 1}@{phase:.0f}deg" for ch, phase in self._channels()),
            )
        self._dispatch_state("running" if start else "idle")

    def _stop(self) -> None:
        if self._hdwf is None:
            return
        lib = self._device_block.lib
        with self._device_block.lock:
            for ch, _ in self._channels():
                lib.FDwfAnalogOutConfigure(self._hdwf, c_int(ch), c_int(0))
        self._running = False
        self._dispatch_state("idle")

    def _dispatch_state(self, state: str) -> None:
        self._output_state.dispatch([np.array([state])])
        if self.parent_graph is not None:
            self.parent_graph.flush_outputs()

    # ── Ports / properties ─────────────────────────────────────────────────

    def _on_enabled(self, values: list[PortValue]):
        enabled = bool(values[0][0])
        if enabled == self._running:
            return
        if enabled:
            self._apply(start=True)
        else:
            self._stop()

    def set_properties(self, properties: dict[str, PropertyDataType]):
        previous_device_index = self._device_index
        reopen = False

        if "device_index" in properties:
            device_index = int(properties["device_index"])
            reopen = device_index != self._device_index
            self._device_index = device_index
        if "channel" in properties:
            self._channel = 1 if int(properties["channel"]) else 0
        if "function" in properties:
            self._function = str(properties["function"]).strip().lower()
        if "frequency" in properties:
            self._frequency = float(properties["frequency"])
        if "amplitude_v" in properties:
            self._amplitude_v = float(properties["amplitude_v"])
        if "offset_v" in properties:
            self._offset_v = float(properties["offset_v"])
        if "differential" in properties:
            self._differential = bool(properties["differential"])
        if "start_on_activate" in properties:
            self._start_on_activate = bool(properties["start_on_activate"])

        self.properties["device_index"] = self._device_index
        self.properties["channel"] = self._channel
        self.properties["function"] = self._function
        self.properties["frequency"] = self._frequency
        self.properties["amplitude_v"] = self._amplitude_v
        self.properties["offset_v"] = self._offset_v
        self.properties["differential"] = self._differential
        self.properties["start_on_activate"] = self._start_on_activate

        if self.parent_graph is not None:
            self.parent_graph.notify_node_updated(self)

        if self.is_active:
            if reopen and self._hdwf is not None:
                self._device_block.release(previous_device_index)
                self._hdwf = None
            if self._hdwf is None:
                try:
                    self._hdwf = self._device_block.acquire(self._device_index)
                except Exception as exc:
                    logger.error("%s: cannot open Analog Discovery: %s", self.name, exc)
                    self._dispatch_state("error")
                    return
            self._apply(start=self._running or self._start_on_activate)
