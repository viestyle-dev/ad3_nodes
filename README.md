# ad3-nodes

Digilent Analog Discovery 3 instruments as nonoha-engine nodes: a gap-free
scope recorder, an on-device waveform generator, and bench-control nodes for
the positive supply, an open-drain DIO and a UART receiver.

## Requirements

[Digilent WaveForms](https://digilent.com/reference/software/waveforms/waveforms-3/start)
must be installed — it ships the `dwf` runtime this plugin calls. There is
nothing to pip-install: the plugin binds the library with `ctypes`, exactly as
the SDK's own Python samples do, so it adds no dependency beyond numpy.

The library is found at `dwf.dll` (Windows), `/Library/Frameworks/dwf.framework/dwf`
or `/Applications/WaveForms.app/Contents/Frameworks/dwf.framework/dwf` (macOS;
recent installers only ship the latter), or `libdwf.so` (Linux).

**An Analog Discovery accepts one owner at a time.** Close the WaveForms
application before running a graph, or the nodes fail with
`Devices are busy, used by other applications`. All nodes in one engine share
the handle, so the supply, reset and UART nodes can run alongside the scope and
generator in the same flow.

## Nodes

### `Ad3Scope`

Continuously records scope channels into the graph.

Uses the SDK's *record* acquisition mode — the only mode that streams without
gaps. Record mode also reports how many samples the device itself had to drop,
which is what separates "the device under test glitched" from "the measurement
rig could not keep up".

Scope inputs are true differential pairs, so a bridge-tied speaker output
connects directly: hot to `1+`, cold to `1-`.

| Property | Default | Description |
| --- | --- | --- |
| `device_index` | `-1` | Device to open; -1 is the first available |
| `sample_rate` | `96000.0` | Acquisition rate in Hz |
| `channels` | `2` | Channels to record (1 or 2) |
| `range_v` | `5.0` | Input range in volts peak-to-peak |
| `pack_size` | `1024` | Samples per dispatched block |

| Output | Values | Description |
| --- | --- | --- |
| `series` | `ch1`, `ch2` (f32) | Captured samples in volts, one value per enabled channel |
| `acq_status` | `lost`, `corrupted`, `dropped` (u32) | Cumulative sample accounting |

`lost` and `corrupted` come from the device; `dropped` counts blocks this node
could not hand to the event loop. **All three must be zero for a result to
mean anything** — treat a non-zero count as an inconclusive measurement rather
than as a fault of the device under test.

Changing `channels` replaces the `series` port, so re-check its edges after
editing that property.

### `Ad3WaveGen`

Drives the waveform generator as a reference signal.

The waveform runs on the device's own synthesizer — nothing is streamed from
the host — so the output is continuous by construction regardless of host load.
That is what makes it usable as a reference for end-to-end capture tests:
any discontinuity observed downstream came from the device under test, not
from here.

| Property | Default | Description |
| --- | --- | --- |
| `device_index` | `-1` | Device to open; -1 is the first available |
| `channel` | `0` | Primary output (0 = W1, 1 = W2) |
| `function` | `"sine"` | `dc`, `sine`, `square`, `triangle`, `ramp_up`, `ramp_down`, `noise` |
| `frequency` | `1000.0` | Hz |
| `amplitude_v` | `0.01` | Volts peak |
| `offset_v` | `0.0` | DC offset in volts |
| `differential` | `false` | Also drive the other channel 180° out of phase |
| `start_on_activate` | `true` | Start generating as soon as the graph runs |

| Port | Direction | Description |
| --- | --- | --- |
| `enabled` | in (bool) | True starts the generator, false returns it to the offset level |
| `state` | out (str) | `idle` / `running` / `error` |

W1 is single-ended, referenced to GND. Set `differential` to drive W2 with the
same waveform in anti-phase, giving a balanced pair for a differential
microphone input (hot = W1, cold = W2).

A stopped generator rests at its offset level rather than going
high-impedance, so the device under test always sees a defined DC level.

> Microphone inputs expect millivolts. Drive them through an external
> attenuator (e.g. a 1000:1 resistive divider) rather than relying on a tiny
> `amplitude_v`.

### `Ad3Supply`

Positive supply V+ at `voltage_v` (AD3: 0.5–5.0 V, clamped); V− stays off.
`enabled` input switches it. Defaults are conservative: it does not switch on
until told (`start_on_activate = false`), and it keeps running when the graph
stops (`off_on_finalize = false`), so stopping a flow never pulls a rail from
under a target that is still powered from elsewhere. The SDK reads back the
V+ voltage only — there is no per-supply current measurement on the AD3.

### `Ad3OpenDrain`

One DIO pin as an open-drain output: `low` holds it low, `pulse` pulls it low
for `pulse_ms` and releases it. Released means high-impedance — the pin is
never driven high, so it can sit on a reset line or a boot strap that the
target pulls up or drives itself. Other DIO pins are left untouched. Put
~1 kΩ in series as insurance against a wrong `dio` index.

### `Ad3UartRx`

UART receiver (8N1) on DIO `rx_dio` at `baud`. Emits each received line on
`line`; with `log_file` set it also appends `HH:MM:SS.mmm <line>` to that file
(relative paths go to the graph's data folder), so a target's boot log can be
read afterwards without a viewer attached.

Using the digital I/O and the UART at the same time on one AD3 is expected to
work (they use different pins); confirm on hardware when a flow first uses
both.

## Design notes

**One handle, two instruments.** An Analog Discovery exposes its scope and its
generator through a single device handle, so the nodes cannot each open the
device independently. `Ad3DeviceBlock` owns one handle per device index and
hands it out by reference count; the node factories inject the shared block.
Every SDK call is serialized
through the block's lock, because the scope reader thread runs concurrently with
property edits arriving on the engine's event loop.

**Auto-configure is off.** The block sets `FDwfDeviceAutoConfigureSet(0)` and
configures each instrument explicitly. With auto-configure on, editing a
generator property would silently restart the running acquisition — a glitch
injected by the measurement itself.

**Blocking calls stay off the event loop.** The scope reader is a plain thread
(the dwf API is blocking); it hands packed blocks to the loop with
`call_soon_threadsafe`, and an asyncio task dispatches them. `__init__` never
touches hardware — the device is opened in `activate()`, as the engine's node
lifecycle requires.

## Installation

Add the plugin to a nonoha-engine workspace (e.g. as `plugins/ad3_nodes`) and
list it in the workspace's plugin manifest and dependencies.

## Tests

From this directory, with a Python environment that has nonoha-engine (for
`nonoha_core`) installed:

```sh
python -m pytest tests -q
```

No hardware needed. The tests build and configure nodes (legal because
`__init__` is side-effect-free), check that every port is documented, and verify
the mirrored SDK constants against the real `dwfconstants.py` when WaveForms is
installed — a wrong constant here would fail silently at runtime.
