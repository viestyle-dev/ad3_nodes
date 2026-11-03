"""Minimal ctypes binding for the Digilent WaveForms SDK (``dwf``).

Deliberately dependency-free: the WaveForms installer already ships the shared
library (``dwf.dll`` on Windows, the ``dwf`` framework on macOS, ``libdwf.so``
on Linux), so there is nothing to pip-install. Only the constants and entry
points actually used by this plugin are declared here — the full constant set
lives in ``<WaveFormsSDK>/samples/py/dwfconstants.py``.
"""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import byref, c_double, c_int, create_string_buffer

# ── Constants (mirrors dwfconstants.py) ────────────────────────────────────

HDWF_NONE = 0

# Acquisition modes — FDwfAnalogInAcquisitionModeSet
ACQMODE_SINGLE = 0
ACQMODE_SCAN_SHIFT = 1
ACQMODE_SCAN_SCREEN = 2
ACQMODE_RECORD = 3

# Instrument states — FDwfAnalogInStatus
DWF_STATE_READY = 0
DWF_STATE_ARMED = 1
DWF_STATE_DONE = 2
DWF_STATE_RUNNING = 3
DWF_STATE_CONFIG = 4
DWF_STATE_PREFILL = 5
DWF_STATE_NOT_DONE = 6
DWF_STATE_WAIT = 7

#: States that mean "the acquisition has not produced samples yet".
DWF_STATES_NOT_STARTED = (DWF_STATE_CONFIG, DWF_STATE_PREFILL, DWF_STATE_ARMED)

# Analog out
ANALOG_OUT_NODE_CARRIER = 0
DWF_ANALOG_OUT_IDLE_DISABLE = 0
DWF_ANALOG_OUT_IDLE_OFFSET = 1
DWF_ANALOG_OUT_IDLE_INITIAL = 2

#: Waveform shapes accepted by the ``function`` property of Ad3WaveGen.
FUNCTIONS: dict[str, int] = {
    "dc": 0,
    "sine": 1,
    "square": 2,
    "triangle": 3,
    "ramp_up": 4,
    "ramp_down": 5,
    "noise": 6,
}

# Analog in channel filters
FILTER_DECIMATE = 0
FILTER_AVERAGE = 1
FILTER_MIN_MAX = 2

# FDwfParamSet
DWF_PARAM_ON_CLOSE = 4  # 0 continue, 1 stop, 2 shutdown

#: Passed to FDwfDeviceOpen to mean "first available device".
DEVICE_INDEX_FIRST = -1

# Analog IO (supplies) — FDwfAnalogIOChannelNode*. On the AD2/AD3 channel 0 is
# the positive supply and channel 1 the negative one; node 0 is enable, node 1
# is voltage (set value and readback).
ANALOG_IO_CH_VPLUS = 0
ANALOG_IO_CH_VMINUS = 1
ANALOG_IO_NODE_ENABLE = 0
ANALOG_IO_NODE_VOLTAGE = 1
#: Settable range of the AD3 positive supply, in volts.
VPLUS_MIN_V = 0.5
VPLUS_MAX_V = 5.0

#: macOS locations of the dwf framework: the system-wide install, then the copy
#: bundled inside WaveForms.app (newer installers only ship the latter).
_DARWIN_LIBRARY_PATHS = (
    "/Library/Frameworks/dwf.framework/dwf",
    "/Applications/WaveForms.app/Contents/Frameworks/dwf.framework/dwf",
)


class DwfError(RuntimeError):
    """A WaveForms SDK call returned failure."""


_lib_lock = threading.Lock()
_lib: ctypes.CDLL | None = None


def load_dwf() -> ctypes.CDLL:
    """Load (once) and return the WaveForms shared library.

    Raises:
        DwfError: if WaveForms is not installed on this machine.
    """
    global _lib
    with _lib_lock:
        if _lib is not None:
            return _lib
        try:
            if sys.platform.startswith("win"):
                _lib = ctypes.cdll.dwf
            elif sys.platform.startswith("darwin"):
                _lib = _load_first(_DARWIN_LIBRARY_PATHS)
            else:
                _lib = ctypes.cdll.LoadLibrary("libdwf.so")
        except OSError as exc:
            raise DwfError(
                "WaveForms SDK library not found. Install Digilent WaveForms "
                "(https://digilent.com/reference/software/waveforms/waveforms-3/start)."
            ) from exc
    return _lib


def _load_first(paths: tuple[str, ...]) -> ctypes.CDLL:
    """Load the first library in *paths* that exists; raise OSError if none does."""
    last_exc: OSError | None = None
    for path in paths:
        try:
            return ctypes.cdll.LoadLibrary(path)
        except OSError as exc:
            last_exc = exc
    raise last_exc or OSError("no candidate paths")


def last_error(lib: ctypes.CDLL) -> str:
    """Return the SDK's description of the most recent failure."""
    buf = create_string_buffer(512)
    lib.FDwfGetLastErrorMsg(buf)
    return buf.value.decode(errors="replace").strip()


def check(lib: ctypes.CDLL, result: int, what: str) -> None:
    """Raise :class:`DwfError` when a dwf call reports failure (returns 0)."""
    if result == 0:
        raise DwfError(f"{what} failed: {last_error(lib)}")


def dwf_version(lib: ctypes.CDLL) -> str:
    """Return the installed WaveForms runtime version, e.g. ``"3.24.2"``."""
    buf = create_string_buffer(32)
    lib.FDwfGetVersion(buf)
    return buf.value.decode(errors="replace")


def open_device(lib: ctypes.CDLL, device_index: int = DEVICE_INDEX_FIRST) -> int:
    """Open an Analog Discovery device and return its handle.

    ``device_index`` of -1 selects the first available device.
    """
    hdwf = c_int()
    lib.FDwfDeviceOpen(c_int(device_index), byref(hdwf))
    if hdwf.value == HDWF_NONE:
        raise DwfError(f"Could not open Analog Discovery device: {last_error(lib)}")
    return hdwf.value


def close_device(lib: ctypes.CDLL, hdwf: int) -> None:
    """Close a single device handle, ignoring failures during teardown."""
    try:
        lib.FDwfDeviceClose(c_int(hdwf))
    except Exception:  # pragma: no cover — teardown must never raise
        pass


def set_on_close_behaviour(lib: ctypes.CDLL, mode: int) -> None:
    """Choose what the hardware does when the handle closes (see DWF_PARAM_ON_CLOSE)."""
    lib.FDwfParamSet(c_int(DWF_PARAM_ON_CLOSE), c_int(mode))


def as_double(value: float) -> c_double:
    """Wrap a Python float for the SDK — dwf takes doubles by value, never ints."""
    return c_double(float(value))
