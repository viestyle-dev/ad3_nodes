"""Node-level tests for the Analog Discovery nodes.

No hardware required: ``__init__`` is side-effect-free by contract, so nodes
can be built, configured and inspected without a device attached — the device
is only opened in ``activate()``.

Running under pytest also enables nonoha-core's autouse ``_validate_dispatch_shapes``
fixture, so every ``dispatch()`` these tests trigger is checked for the batch
dimension the port system requires.
"""

from __future__ import annotations

import numpy as np
import pytest

import ad3_nodes
from ad3_nodes import dwf_binding as dwf
from ad3_nodes.device_block import Ad3DeviceBlock
from ad3_nodes.scope_nodes import Ad3ScopeNode
from ad3_nodes.wavegen_nodes import Ad3WaveGenNode
from nonoha_core._testing import assert_all_ports_documented
from nonoha_core.node import NodeDoc
from nonoha_core.node_registry import NodeRegistry


@pytest.fixture
def device_block() -> Ad3DeviceBlock:
    return Ad3DeviceBlock()


@pytest.fixture
def scope(device_block) -> Ad3ScopeNode:
    return Ad3ScopeNode("scope", device_block)


@pytest.fixture
def wavegen(device_block) -> Ad3WaveGenNode:
    return Ad3WaveGenNode("wavegen", device_block)


def capture(node, port_name: str) -> list[list[np.ndarray]]:
    """Record everything dispatched from *port_name*."""
    received: list[list[np.ndarray]] = []
    node.outputs[port_name].subscribe(id(received), received.append)
    return received


# ── Plugin registration ────────────────────────────────────────────────────


def test_register_nodes_documents_every_port():
    """nonoha requires every non-property port to be described in describe()."""
    registry = NodeRegistry()
    ad3_nodes.register_nodes(registry)
    assert_all_ports_documented(registry)


def test_register_nodes_registers_all_instruments():
    registry = NodeRegistry()
    ad3_nodes.register_nodes(registry)
    assert set(registry.node_definitions) == {
        "Ad3Scope",
        "Ad3WaveGen",
        "Ad3Supply",
        "Ad3OpenDrain",
        "Ad3UartRx",
    }


def test_nodes_share_one_device_block():
    """An Analog Discovery exposes scope and generator through one handle."""
    assert ad3_nodes.get_device_block() is ad3_nodes.get_device_block()


# ── Ad3Scope ───────────────────────────────────────────────────────────────


class TestAd3ScopeNode:
    def test_describe(self):
        doc = Ad3ScopeNode.describe()
        assert isinstance(doc, NodeDoc)
        assert doc.category == "AD3"
        assert set(doc.outputs) == {"series", "acq_status"}

    def test_property_defaults(self, scope):
        assert scope.properties["sample_rate"] == 96000.0
        assert scope.properties["channels"] == 2
        assert scope.properties["range_v"] == 5.0
        assert scope.properties["pack_size"] == 1024
        assert scope.properties["device_index"] == -1

    def test_series_port_has_one_value_per_channel(self, scope):
        assert scope.outputs["series"].value_name_list == ["ch1", "ch2"]

    def test_channel_count_change_replaces_series_port(self, scope):
        old_port = scope.outputs["series"]
        scope.set_properties({"channels": 1})
        new_port = scope.outputs["series"]
        assert new_port is not old_port
        assert new_port.value_name_list == ["ch1"]

    def test_channel_count_is_clamped_to_available_hardware(self, scope):
        """An Analog Discovery has two scope channels; asking for more is a typo."""
        scope.set_properties({"channels": 7})
        assert scope.properties["channels"] == 2
        scope.set_properties({"channels": 0})
        assert scope.properties["channels"] == 1

    def test_pack_size_cannot_be_zero(self, scope):
        scope.set_properties({"pack_size": 0})
        assert scope.properties["pack_size"] == 1

    def test_acq_status_separates_device_loss_from_host_loss(self, scope):
        """Telling these apart is the whole point: a rig fault is not a DUT fault."""
        assert scope.outputs["acq_status"].value_name_list == [
            "lost",
            "corrupted",
            "dropped",
        ]

    def test_unchanged_properties_are_preserved(self, scope):
        scope.set_properties({"sample_rate": 48000.0})
        assert scope.properties["sample_rate"] == 48000.0
        assert scope.properties["channels"] == 2
        assert scope.properties["range_v"] == 5.0


# ── Ad3WaveGen ─────────────────────────────────────────────────────────────


class TestAd3WaveGenNode:
    def test_describe(self):
        doc = Ad3WaveGenNode.describe()
        assert isinstance(doc, NodeDoc)
        assert doc.category == "AD3"
        assert "enabled" in doc.inputs
        assert "state" in doc.outputs

    def test_property_defaults(self, wavegen):
        assert wavegen.properties["function"] == "sine"
        assert wavegen.properties["frequency"] == 1000.0
        assert wavegen.properties["amplitude_v"] == 0.01
        assert wavegen.properties["differential"] is False
        assert wavegen.properties["start_on_activate"] is True

    def test_single_ended_drives_only_the_primary_channel(self, wavegen):
        assert wavegen._channels() == [(0, 0.0)]

    def test_differential_drives_the_other_channel_in_antiphase(self, wavegen):
        """Balanced mic input: hot on W1, cold on W2 180 degrees out of phase."""
        wavegen.set_properties({"differential": True})
        assert wavegen._channels() == [(0, 0.0), (1, 180.0)]

    def test_differential_respects_the_chosen_primary_channel(self, wavegen):
        wavegen.set_properties({"channel": 1, "differential": True})
        assert wavegen._channels() == [(1, 0.0), (0, 180.0)]

    def test_function_name_is_normalised(self, wavegen):
        wavegen.set_properties({"function": "  SINE "})
        assert wavegen.properties["function"] == "sine"

    def test_unknown_function_reports_error_state(self, wavegen):
        """A typo in the waveform name must surface, not silently do nothing."""
        states = capture(wavegen, "state")
        wavegen._hdwf = 1  # pretend a device is open; _apply fails before using it
        wavegen.set_properties({"function": "sawtooth"})
        wavegen._apply(start=True)
        assert [str(values[0][0]) for values in states] == ["error"]

    def test_state_is_dispatched_as_a_string_array(self, wavegen):
        states = capture(wavegen, "state")
        wavegen._dispatch_state("idle")
        assert len(states) == 1
        assert str(states[0][0][0]) == "idle"

    def test_enabled_input_is_ignored_when_already_in_that_state(self, wavegen):
        """Re-asserting the current state must not reconfigure the hardware."""
        states = capture(wavegen, "state")
        wavegen._on_enabled([np.array([False])])
        assert states == []


# ── Vendored SDK surface ───────────────────────────────────────────────────


def test_every_documented_function_name_maps_to_a_code():
    assert dwf.FUNCTIONS["sine"] == 1
    assert set(dwf.FUNCTIONS) == {
        "dc",
        "sine",
        "square",
        "triangle",
        "ramp_up",
        "ramp_down",
        "noise",
    }


def test_record_is_the_streaming_acquisition_mode():
    """Only record mode streams continuously; the others capture a finite frame."""
    assert dwf.ACQMODE_RECORD == 3
