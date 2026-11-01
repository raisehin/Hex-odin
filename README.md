# hex (Odin)

A from-scratch Odin implementation of [Hex](https://anoma.ly/hex)'s dictation
core and overlay, replacing the Rust application while keeping the C++ inference
stack (`transcribe.cpp` / `ggml` / Vulkan) linked from Odin.

Target: **no Rust in the build.** `C:\Users\rhais\Documents\hex` is the reference
implementation, read-only here.

## Status

| Piece | State |
|---|---|
| CLI (`devices`, `record`, `transcribe`, `models`, `status`, `app`) | working |
| Capture via miniaudio (WASAPI backend) → mono f32 16 kHz → WAV | working |
| 12 kLOC-free: ~1.7k lines of Odin so far | — |
| HUD: port of Rune's **Dot scope** (112x26 pt plate, 43x9 dots @2.6 pt, shadow/rim/glow, braid animation, 2x supersampled) | working |
| Global hotkeys: `Ctrl+Space` dictate, `Ctrl+Alt+Q` quit | working |
| Clipboard paste + restore | working - transcript goes on the clipboard, a paste chord is sent, and the old clipboard is put back 500 ms later unless you copied something in the meantime. Every dictation pastes. Only plain text is put back afterwards, so a copied image, file or rich text is replaced by the transcript and the log names the formats that were dropped (`DataObject`, `Ole Private Data`, `EnterpriseDataProtectionId` and the other clipboard bookkeeping formats are recognised as metadata, not content, so copying from Word or Edge still restores the old text). Typing is only used with `hex paste type`. Verified end to end in Notepad: text arrived, clipboard returned to `ORIGINAL-123`. `hex paste <ctrl_v\|ctrl_shift_v\|shift_insert\|type>` |
| Model registry + `curl` download + SHA-256 verify (`certutil`) | working |
| C++ shim ABI (`hex_model_init` / `hex_transcribe` / `hex_list_devices`) | DLL links ggml + transcribe.cpp statically, loads via `core:dynlib` |
| Escape cancel | working: Esc while recording discards it; Esc while transcribing cancels so nothing pastes (registered only during a dictation, so Escape stays free elsewhere) |
| Silero VAD | working: `silero-vad-v6.2-F32.gguf` gates the final transcription, so silence never reaches the ASR (no more invented "Sorry") |
| Log file | working: `%APPDATA%\hex\logs\odin.log`, append-only with timestamps, because the desktop launch has no console |
| Mic on demand | working: the device opens when the hotkey starts a dictation and is released the moment it ends, so Windows never shows the mic in use while hex is idle (verified against the CapabilityAccessManager mic timestamps). The ring still holds the last 2 minutes |
| Model idle release | working: the ASR session is freed after 3 minutes without a dictation (`HEX_MODEL_IDLE_SECS` overrides). Resident drops from 411 MB private / 1.1 GB working set to 32 MB / 22 MB; the next dictation reloads it in about a second |
| Idle overhead | 0.2% of a core and 21 MB private with the model unloaded. The 24 fps HUD timer stops itself when hidden, the capture pump thread exists only during a dictation, and the process sits in Windows EcoQoS while idle |
| Thread cap | `hex threads <n>` caps ggml's CPU workers (`0` = every usable core). Capped to 4 on this 4c/8t machine: peak package power roughly halves while latency moves by a few percent |
| Real inference | **working**: `hex transcribe samples/fleurs-en.wav` -> "However, due to the slow communication channels, styles in the West could lag behind by 25 to 30 years." (CPU backend, 2.8 s incl. model mmap) |

Shared data root: `%APPDATA%\hex`, so this build and the old Rust install see
the same models.

The registry is deliberately short: **Phonon-2.5 English Q8_0** (default) and
**Canary 180M Flash**. `scripts/gen_models.py` filters Handy's catalog down to
those two, so `hex models list` cannot drift back to a menu of 70 checkpoints.

Log lines from the C++ side are named after the architecture, not the file:
Phonon-2.5 is a Parakeet-family checkpoint, so `parakeet: weights memory-mapped`
is expected. The shim prints `[hex] model file: ...` right before it so the
actual GGUF is never in doubt.

## Overhead

Measured on this laptop with the model unloaded (dictation itself is a separate
burst):

| state | before | after |
|---|---|---|
| idle, HUD hidden | 1.1% of a core, 50 MB private | **0.2% of a core, 21 MB private** |
| recording (HUD visible) | 12.0% of a core | **5.0% of a core** |
| wakeups while idle | 60 Hz timer + 100 Hz pump poll | one 5 s housekeeping tick |

What changed: the HUD renders at 24 fps instead of 60, its plate/rim/shadow are
baked once so the per-frame pass only does glow + dots, and the window timer is
killed while hidden. The capture pump starts with the hotkey and is joined on
release, so idle costs no polling. The ring is 2 minutes, not 10.

Inference knob: `hex threads <n>` caps ggml's workers (peak CPU/heat; 0 = auto).
Transcription on FLEURS English with 4 threads takes 3.9 s end to end including
model load; 8 threads only takes 3.0 s, which is why the cap is nearly free.
Memory returns to baseline within a few seconds of the idle release, so a
dictation costs ~400 MB for its duration and nothing while you are not using it.

## Build

```powershell
# Odin app (~5 s)
.\build.ps1                 # debug
.\build.ps1 -Release

# Heavy half, built once: ggml + transcribe.cpp (CPU backend, no Vulkan SDK needed)
cmake -S native/third_party/transcribe.cpp -B native/build-transcribe -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF
cmake --build native/build-transcribe --config Release --parallel

# Shim, linked against those static libs, copied next to hex.exe
.\build.ps1 -Release -Native
```

Vulkan is a flag away (`-DGGML_VULKAN=ON`) but needs the Vulkan SDK, which is
not installed here; CPU is the default.

If `native/build-transcribe/.../transcribe.lib` is missing, the shim builds
anyway (stub) so the app stays runnable.

Odin lives at `C:\Users\rhais\Downloads\odin-windows-amd64-dev-2026-10`
(`dist\odin.exe`, dev-2026-10-nightly).

## Use
 ```powershell
python scripts/gen_models.py <handy>\src-tauri\src\catalog\catalog.json
``` Adding or updating a model is data, not code - append a
`Model` literal to `MODELS` in `src/models.odin`. transcribe.cpp picks the
architecture out of the GGUF itself, so no rebuild is involved.

### Phonon-2.5 English

Phonon-2.5 English (Fermion Research, 0.6B TDT) is in the registry as
`phonon25_en` and is the default. It is the one model that cannot be downloaded
as-is: upstream ships a packed 164 MB container (`phonon-2.5-en.bps.tar.zst`),
and public GGUF conversions use incompatible tensor names, so it has to be
converted once:

```powershell
./scripts/fetch-phonon25.ps1 -Install
```

That pins and verifies the upstream artifact, expands it to a dense F32 GGUF
with transcribe.cpp's own converter (driven by
`scripts/convert-phonon25.py`, which reuses the upstream Fermion reader with the
2.5 hashes), quantizes with `transcribe-quantize`, and drops
`phonon-2.5-en-Q8_0.gguf` into `%APPDATA%\hex\models`.

Python/torch appear only in that one-time conversion; loading, dictation and the
GUI stay Odin + C++. Nothing on the runtime path touches Python — torch is not
installed now, and dictation with `phonon25_en` works without it.

Default: **Phonon-2.5 English Q8_0** (`phonon25_en`, 705 MB, English). Switch
with `hex models use <id>`; a config pointing at a missing model falls back to
the default instead of failing at startup. Phonon-2 was removed — same 0.6B
size class, 2.5 is the newer English model and no longer drops spaces
(`by25 -30` became `by 25 to 30`).

## Layout

```
src/main.odin        CLI dispatch
src/capture.odin     miniaudio capture, 2-minute lock-free ring, level meter
src/wav.odin         RIFF writer, 16-bit reader with mono downmix + resample
src/transcribe.odin  dynlib shim loader + registry→path resolution
src/models.odin      port of transcription_models.rs (id, repo, rev, sha256, langs)
src/config.odin      %APPDATA%\hex\odin-config
src/paste.odin       SendInput Unicode
src/hud.odin         Dot scope rasterizer (port of rune_hud.wgsl), overlay window, hotkeys
native/              C ABI shim + CMakeLists (FetchContent at the pinned rev)
```

## Mapping from the Rust tree

| Rust | Odin |
|---|---|
| `src/windows_audio.rs`, `cpal` | `capture.odin` (vendor:miniaudio) |
| `src/windows_paste.rs` | `paste.odin` |
| `src/transcription_models.rs` | `models.odin` (data only) |
| `src/gguf_session.rs` | `native/transcribe_shim.cpp` |
| `src/rune_hud.rs` + `rune_hud.wgsl` | `hud.odin` (DIB for now) |
| `src/rune_ui/*` (56k lines) | **not ported yet** — window/pane system is next |
| `src/local_api.rs` | not ported yet |
| `src/updater.rs`, release protocol | not ported yet |

## Next steps, in order