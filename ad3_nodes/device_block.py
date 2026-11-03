"""Shared Analog Discovery device handle.

An Analog Discovery exposes its scope and its waveform generator through a
*single* device handle, so ``Ad3Scope`` and ``Ad3WaveGen`` cannot each open the
device independently. This block owns one handle per device index and hands it
out by reference count; the node factories inject the shared block.
"""

from __future__ import annotations

import logging
import threading

from . import dwf_binding as dwf

logger = logging.getLogger(__name__)


class Ad3DeviceBlock:
    """Reference-counted owner of Analog Discovery device handles.

    All SDK calls made through :attr:`lock` are serialized: the WaveForms
    runtime tolerates concurrent access to different instruments on one handle
    far less reliably than it tolerates serialized access, and the scope reader
    thread runs concurrently with property edits from the engine's event loop.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handles: dict[int, int] = {}
        self._refcounts: dict[int, int] = {}

    @property
    def lock(self) -> threading.RLock:
        """Serializes every SDK call against this block's handles."""
        return self._lock

    def acquire(self, device_index: int) -> int:
        """Open (or reuse) the device and return its handle, bumping the refcount."""
        lib = dwf.load_dwf()
        with self._lock:
            handle = self._handles.get(device_index)
            if handle is None:
                # Keep the instruments running when the last handle closes, so a
                # graph restart does not glitch the reference signal.
                dwf.set_on_close_behaviour(lib, 0)
                handle = dwf.open_device(lib, device_index)
                self._handles[device_index] = handle
                self._refcounts[device_index] = 0
                # Explicit configuration only: an auto-configure would restart the
                # running acquisition every time a wavegen property is edited.
                lib.FDwfDeviceAutoConfigureSet(handle, 0)
                logger.info(
                    "Ad3DeviceBlock: opened device %s (WaveForms %s)",
                    device_index,
                    dwf.dwf_version(lib),
                )
            self._refcounts[device_index] += 1
            return handle

    def release(self, device_index: int) -> None:
        """Drop a reference; close the handle once nothing holds it."""
        lib = dwf.load_dwf()
        with self._lock:
            if device_index not in self._refcounts:
                return
            self._refcounts[device_index] -= 1
            if self._refcounts[device_index] > 0:
                return
            handle = self._handles.pop(device_index, None)
            self._refcounts.pop(device_index, None)
            if handle is not None:
                dwf.close_device(lib, handle)
                logger.info("Ad3DeviceBlock: closed device %s", device_index)

    @property
    def lib(self):
        """The loaded WaveForms shared library."""
        return dwf.load_dwf()

    def finalize(self) -> None:
        """Close every outstanding handle (plugin shutdown)."""
        lib = dwf.load_dwf()
        with self._lock:
            for handle in self._handles.values():
                dwf.close_device(lib, handle)
            self._handles.clear()
            self._refcounts.clear()
