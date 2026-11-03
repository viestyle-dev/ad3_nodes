"""Tests for the AD3 supply, open-drain and UART nodes, against a fake SDK.

The fake records every call and keeps the digital I/O registers, so the
open-drain behaviour (never drive high, keep other pins untouched) can be
checked without a device attached.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from ad3_nodes import dwf_binding as dwf
from ad3_nodes.device_block import Ad3DeviceBlock
from ad3_nodes.io_nodes import Ad3OpenDrainNode, Ad3SupplyNode, Ad3UartRxNode
from nonoha_core.node import NodeDoc


class FakeDwf:
    """Just enough of the dwf library for the I/O nodes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple]] = []
        self.dio_enable = 0
        self.dio_output = 0

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == "FDwfDigitalIOOutputEnableGet":
                args[1]._obj.value = self.dio_enable
            elif name == "FDwfDigitalIOOutputGet":
                args[1]._obj.value = self.dio_output
            elif name == "FDwfDigitalIOOutputEnableSet":
                self.dio_enable = args[1].value
            elif name == "FDwfDigitalIOOutputSet":
                self.dio_output = args[1].value
            elif name == "FDwfAnalogIOChannelNodeStatus":
                args[3]._obj.value = 1.8
            return 1

        return call

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]


@pytest.fixture
def fake(monkeypatch) -> FakeDwf:
    lib = FakeDwf()
    monkeypatch.setattr(dwf, "load_dwf", lambda: lib)
    return lib


@pytest.fixture
def device_block() -> Ad3DeviceBlock:
    return Ad3DeviceBlock()


def capture(node, port_name: str) -> list[list[np.ndarray]]:
    received: list[list[np.ndarray]] = []
    node.outputs[port_name].subscribe(id(received), received.append)
    return received


def states(received) -> list[str]:
    return [str(values[0][0]) for values in received]


# ── Ad3Supply ──────────────────────────────────────────────────────────────


class TestAd3SupplyNode:
    def test_describe(self):
        doc = Ad3SupplyNode.describe()
        assert isinstance(doc, NodeDoc)
        assert doc.category == "AD3"
        assert set(doc.inputs) == {"enabled"}
        assert set(doc.outputs) == {"state"}

    def test_defaults_are_safe(self, device_block):
        """1.8 V, and nothing switches on by itself or off behind a powered target."""
        node = Ad3SupplyNode("supply", device_block)
        assert node.properties["voltage_v"] == 1.8
        assert node.properties["start_on_activate"] is False
        assert node.properties["off_on_finalize"] is False

    def test_voltage_is_clamped_to_the_ad3_range(self, device_block):
        node = Ad3SupplyNode("supply", device_block)
        node.set_properties({"voltage_v": 9.0})
        assert node._clamped_voltage() == dwf.VPLUS_MAX_V
        node.set_properties({"voltage_v": 0.1})
        assert node._clamped_voltage() == dwf.VPLUS_MIN_V

    def test_switching_on_sets_vplus_and_keeps_vminus_off(self, fake, device_block):
        node = Ad3SupplyNode("supply", device_block)
        out = capture(node, "state")
        node._hdwf = 1
        node._apply(True)

        sets = [args for name, args in fake.calls if name == "FDwfAnalogIOChannelNodeSet"]
        as_tuple = [(a[1].value, a[2].value, a[3].value) for a in sets]
        assert (dwf.ANALOG_IO_CH_VPLUS, dwf.ANALOG_IO_NODE_VOLTAGE, 1.8) in as_tuple
        assert (dwf.ANALOG_IO_CH_VPLUS, dwf.ANALOG_IO_NODE_ENABLE, 1.0) in as_tuple
        assert (dwf.ANALOG_IO_CH_VMINUS, dwf.ANALOG_IO_NODE_ENABLE, 0.0) in as_tuple
        assert "FDwfAnalogIOConfigure" in fake.names()
        assert states(out) == ["on 1.800 V"]

    def test_switching_off(self, fake, device_block):
        node = Ad3SupplyNode("supply", device_block)
        out = capture(node, "state")
        node._hdwf = 1
        node._apply(False)
        assert states(out) == ["off"]


# ── Ad3OpenDrain ───────────────────────────────────────────────────────────


class TestAd3OpenDrainNode:
    def test_describe(self):
        doc = Ad3OpenDrainNode.describe()
        assert set(doc.inputs) == {"low", "pulse"}
        assert set(doc.outputs) == {"state"}

    def test_low_enables_the_pin_with_a_zero_latch(self, fake, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node._hdwf = 1
        fake.dio_output = 0xFFFF  # whatever another user left in the latch
        node._on_low([np.array([True])])
        assert fake.dio_enable & 0x1
        assert not fake.dio_output & 0x1

    def test_release_disables_the_pin_instead_of_driving_high(self, fake, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node._hdwf = 1
        node._on_low([np.array([True])])
        node._on_low([np.array([False])])
        assert not fake.dio_enable & 0x1
        assert not fake.dio_output & 0x1

    def test_other_pins_are_left_alone(self, fake, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node.set_properties({"dio": 3})
        node._hdwf = 1
        fake.dio_enable = 0b0100_0000
        node._on_low([np.array([True])])
        assert fake.dio_enable == 0b0100_1000
        node._on_low([np.array([False])])
        assert fake.dio_enable == 0b0100_0000

    def test_dio_is_clamped(self, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node.set_properties({"dio": 99})
        assert node.properties["dio"] == 15

    def test_pulse_goes_low_then_releases(self, fake, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node.set_properties({"pulse_ms": 5.0})
        out = capture(node, "state")
        node._hdwf = 1

        async def run():
            node._on_pulse([np.array([True])])
            assert fake.dio_enable & 0x1
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert states(out) == ["low", "released"]
        assert not fake.dio_enable & 0x1

    def test_pulse_does_not_release_a_held_pin(self, fake, device_block):
        node = Ad3OpenDrainNode("run", device_block)
        node.set_properties({"pulse_ms": 5.0})
        node._hdwf = 1

        async def run():
            node._on_low([np.array([True])])
            node._on_pulse([np.array([True])])
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert fake.dio_enable & 0x1


# ── Ad3UartRx ──────────────────────────────────────────────────────────────


class TestAd3UartRxNode:
    def test_describe(self):
        doc = Ad3UartRxNode.describe()
        assert set(doc.outputs) == {"line", "state"}

    def test_property_defaults(self, device_block):
        node = Ad3UartRxNode("uart", device_block)
        assert node.properties["rx_dio"] == 1
        assert node.properties["baud"] == 115200.0
        assert node.properties["log_file"] == ""

    def test_configure_selects_pin_rate_and_8n1(self, fake, device_block):
        node = Ad3UartRxNode("uart", device_block)
        node.set_properties({"rx_dio": 2, "baud": 9600.0})
        node._configure(fake, 1)
        calls = dict((name, args) for name, args in fake.calls)
        assert calls["FDwfDigitalUartRxSet"][1].value == 2
        assert calls["FDwfDigitalUartRateSet"][1].value == 9600.0
        assert calls["FDwfDigitalUartBitsSet"][1].value == 8
        assert calls["FDwfDigitalUartParitySet"][1].value == 0
