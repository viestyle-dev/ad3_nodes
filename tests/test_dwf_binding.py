"""Tests for the vendored WaveForms constants.

The constants in :mod:`ad3_nodes.dwf_binding` are copied from the SDK's
``dwfconstants.py``. Copies drift, and a wrong constant here fails silently at
runtime — an acquisition mode that quietly is not "record" would look like a
device fault. So when the SDK is installed, check the copy against the source.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from ad3_nodes import dwf_binding as dwf

#: Where the WaveForms installer puts its Python samples on each platform.
_SDK_SAMPLE_DIRS = [
    Path(r"C:\Program Files\Digilent\WaveFormsSDK\samples\py"),
    Path(r"C:\Program Files (x86)\Digilent\WaveFormsSDK\samples\py"),
    Path("/Applications/WaveForms.app/Contents/Resources/SDK/samples/py"),
    Path("/usr/share/digilent/waveforms/samples/py"),
]

#: Vendored name -> name in dwfconstants.py
_MIRRORED = {
    "ACQMODE_SINGLE": "acqmodeSingle",
    "ACQMODE_SCAN_SHIFT": "acqmodeScanShift",
    "ACQMODE_SCAN_SCREEN": "acqmodeScanScreen",
    "ACQMODE_RECORD": "acqmodeRecord",
    "DWF_STATE_READY": "DwfStateReady",
    "DWF_STATE_ARMED": "DwfStateArmed",
    "DWF_STATE_DONE": "DwfStateDone",
    "DWF_STATE_RUNNING": "DwfStateRunning",
    "DWF_STATE_CONFIG": "DwfStateConfig",
    "DWF_STATE_PREFILL": "DwfStatePrefill",
    "DWF_STATE_NOT_DONE": "DwfStateNotDone",
    "DWF_STATE_WAIT": "DwfStateWait",
    "ANALOG_OUT_NODE_CARRIER": "AnalogOutNodeCarrier",
    "DWF_ANALOG_OUT_IDLE_DISABLE": "DwfAnalogOutIdleDisable",
    "DWF_ANALOG_OUT_IDLE_OFFSET": "DwfAnalogOutIdleOffset",
    "DWF_ANALOG_OUT_IDLE_INITIAL": "DwfAnalogOutIdleInitial",
    "FILTER_DECIMATE": "filterDecimate",
    "FILTER_AVERAGE": "filterAverage",
    "FILTER_MIN_MAX": "filterMinMax",
    "DWF_PARAM_ON_CLOSE": "DwfParamOnClose",
}

_FUNCTION_NAMES = {
    "dc": "funcDC",
    "sine": "funcSine",
    "square": "funcSquare",
    "triangle": "funcTriangle",
    "ramp_up": "funcRampUp",
    "ramp_down": "funcRampDown",
    "noise": "funcNoise",
}


def _find_sdk_constants() -> dict[str, int] | None:
    """Parse the SDK's dwfconstants.py, or return None if it is not installed.

    Parsed with a regex rather than imported: the SDK module is written for
    ``from ctypes import *`` at module scope and importing it would pull the
    whole ctypes namespace into the test process.
    """
    for directory in _SDK_SAMPLE_DIRS:
        path = directory / "dwfconstants.py"
        if path.is_file():
            source = path.read_text(encoding="utf-8", errors="replace")
            pattern = re.compile(
                r"^(\w+)\s*=\s*c_(?:u?byte|u?int|u?short)\((-?\d+)\)", re.MULTILINE
            )
            return {name: int(value) for name, value in pattern.findall(source)}
    return None


@pytest.fixture(scope="module")
def sdk_constants() -> dict[str, int]:
    constants = _find_sdk_constants()
    if constants is None:
        pytest.skip("WaveForms SDK samples not installed on this machine")
    return constants


@pytest.mark.parametrize("vendored_name,sdk_name", sorted(_MIRRORED.items()))
def test_constant_matches_sdk(sdk_constants, vendored_name, sdk_name):
    assert getattr(dwf, vendored_name) == sdk_constants[sdk_name]


@pytest.mark.parametrize("key,sdk_name", sorted(_FUNCTION_NAMES.items()))
def test_function_code_matches_sdk(sdk_constants, key, sdk_name):
    assert dwf.FUNCTIONS[key] == sdk_constants[sdk_name]


def test_states_not_started_are_the_pre_acquisition_states():
    """Reading data before these clear yields nothing; the reader loop waits."""
    assert set(dwf.DWF_STATES_NOT_STARTED) == {
        dwf.DWF_STATE_CONFIG,
        dwf.DWF_STATE_PREFILL,
        dwf.DWF_STATE_ARMED,
    }


def test_as_double_wraps_python_floats():
    """dwf takes doubles by value; passing a bare int silently corrupts the call."""
    value = dwf.as_double(96000)
    assert value.value == pytest.approx(96000.0)
    assert type(value).__name__ == "c_double"


def test_function_names_are_the_documented_set():
    assert set(dwf.FUNCTIONS) == set(_FUNCTION_NAMES)


@pytest.mark.skipif(
    not sys.platform.startswith("win"), reason="library path checked on Windows only"
)
def test_library_loads_when_waveforms_is_installed():
    """A missing WaveForms install must fail with an actionable message."""
    try:
        lib = dwf.load_dwf()
    except dwf.DwfError as exc:
        assert "WaveForms" in str(exc)
        pytest.skip("WaveForms runtime not installed")
    else:
        assert dwf.dwf_version(lib)
