# hex (Odin)

Hex's dictation core in plain Odin: capture, the Dot-scope overlay, hotkeys,
clipboard paste — and, since this branch, the speech recogniser too.

`phonon/` is a hand-written inference engine for **Phonon-2.5 English**
(Fermion Research, 0.6B TDT / Parakeet family): it reads the shipped
`Q8_0`/`F16` GGUF directly, runs the NeMo log-mel frontend, a 24-block
conformer encoder and the TDT decoder, and does its own matmuls with
**AVX2 / FMA kernels written for Zen 2** — no ggml, no BLAS, no C++ on
the ASR path. The C++ stack (`transcribe.cpp` / `ggml`) is still linked as a
reference oracle and as the path for every other model family.

The claim it has to defend is not "it runs" but "it computes the same thing":
both engines produce **byte-identical transcripts** on the benchmark clips, and
the encoder stages match the reference dumps at 1, 2, 4, 6 and 8 threads.

| Piece | State |
|---|---|
| ASR engine in Odin: GGUF reader, mel frontend, conformer encoder, TDT decoder | working, parity with ggml |
| Q8_0 / F16 / F32 kernels (AVX2 + FMA) | working, tuned for 4C/8T Zen 2 |
| CLI (`devices`, `record`, `transcribe`, `models`, `status`, `engine`, `threads`, `paste`, `app`) | working |
| Capture via miniaudio (WASAPI) → mono f32 16 kHz → WAV, 2-minute ring | working |
| HUD: port of Rune's **Dot scope** (112x26 pt plate, 43x9 dots @2.6 pt, shadow/rim/glow, braid animation, 2x supersampled) | working |
| Global hotkeys `Ctrl+Space` dictate, `Ctrl+Alt+Q` quit; Esc cancels a recording or a transcription | working |
| Clipboard paste + restore | working — transcript onto the clipboard, paste chord sent, old clipboard put back 500 ms later unless you copied something meanwhile. Only plain text is restored, so a copied image/file/rich text is replaced; the log names the formats dropped. `hex paste <ctrl_v\|ctrl_shift_v\|shift_insert\|type>` |
| Silero VAD gate | working — silence never reaches the ASR (no invented "Sorry") |
| Model registry + `curl` + SHA-256 (`certutil`) | working; Phonon-2.5 Q8_0 (default) and Canary 180M Flash |
| Mic on demand, model idle release | working — device opens on the hotkey and closes on release; the ASR session is freed after 3 min idle (`HEX_MODEL_IDLE_SECS`), resident 411 MB → 32 MB private |
| Lines of code | 3.6 k Odin in `src/`, 2.5 k in `phonon/`, 280 C++ in `native/` |

## The kernels

Everything here is in `phonon/qkernels.odin` (+ the dispatch and thread pool in
`phonon/linalg.odin`). The model file keeps 225 of its 697 tensors as `Q8_0`
blocks and 48 as `F16`; the rest are f32. A `Q8_0` tensor is 34 bytes per 32
output-side values (an f16 scale and 32 int8), so the encoder streams about
**600 MB of weights per utterance** on four Zen 2 cores. The first, obvious
port was scalar and sat ~15x off ggml; these kernels close that gap.

**Q8_0 × Q8_0 dot.** Activations are quantised once per frame (`d = amax/127`,
scale stored as **f16**, exactly like `GGML_FP32_TO_FP16`, because keeping `d`
in f32 shifts every block by ~5e-4 relative and the LayerNorms amplify that past
the decoding margin). Then per 32-value block: load the int8 weights and the
int8 activations, `vpmaddubsw` the products (the activation is taken as
*unsigned*, so its sign is folded into the weight with **`vpsignb`** — the
straightforward select-over-negate version compiled to `vpcmpgtb + vpxor +
vpsubb`, three uops on the critical path of every block), `vpmaddwd` to widen
the i16 pairs into i32, convert, and fold the block scale in with a single
`fma`. Four blocks run in independent accumulators, which hides the 13-cycle
madd → cvt → fma chain behind throughput.

**Two frames per weight pass.** `lin_task` visits every output row once per
frame, so the encoder re-reads its weights ~11 times per utterance: 11.7 GFLOP
at 63 GFLOPS is **35 GB/s of row bytes**, i.e. the matmul path saturates DRAM
rather than the FPU. `q8_0_dot_q8_2f` dots two quantised frames against the same
loaded block, halving that traffic. Per frame it accumulates in the same four
interleaved accumulators over the same block quad and reduces in the same tree,
so it cannot move the output: the transcripts stay byte-identical.

**The subsampling front end.** `pre_encode` is 6 convolutions over a
`[mels][frames]` plane, and it used to cost 33 ms of the encoder's 280: every
stage materialised a transposed copy of its input, ran a matmul
with the *wrong* layout (`[oc][9]` against `[pos][k]`), then transposed a
`[pos][ocs]` result back to channel-major. The shapes are small (256 channels,
106 frames), so all three passes were cache-resident copies whose only purpose
was to satisfy the matmul's row-major assumption. Now each conv reads the
existing channel-major plane directly: `conv_first_3x3s2` walks the 9 taps with
the output position innermost, `conv_1x1` runs 8 positions × 4 bias-free
accumulators over the contiguous row (the weight row is sequential, the
cross-channel stride is the only non-sequential access), and the final
`enc.pre_encode.out` projection goes through `lin_rows_par` (Q8_0, 4096 → 1024)
instead of one frame per call. `conv2d` stays a pure convolution on every
dispatch path — folding `relu` into the first conv broke exactly that contract
(the shape test compares against the naive definition) and cost nothing to
keep as its own pass.

**F16 and F32.** 48 tensors ship as f16 and are promoted to f32 once at load
(`promote_f16`), so the hot loops only ever see f32 or int8: `f32_dot_avx` is
four interleaved 8-lane FMA accumulators over 32 values per iteration, and
`half_to_float` is a scalar bit-twiddle that matches ggml's conversion for
normals, subnormals, infinities and NaNs. (F16C would let the f16 tensors stay
shipped as f16 — see the next steps.)

**Threads.** The Odin engine's pool has none of ggml's per-node barriers, so
`threads = 0` means every logical CPU; for the shim path, `worker_threads`
clamps to three quarters of the logical count, because ggml's pool spins
between nodes and hands an SMT sibling work that starves its physical core.

Measured on this laptop (4C/8T Zen 2, 16 kHz mono, one fresh process per run;
`bench-kernels/src` times the kernels in isolation):

| threads | 2 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| 1 s clip, infer ms | 631 | 418 | 374 | **328–353** | 376 | 459–822 |

That sweep is why the shim cap is 3/4 of the logical count: 8 threads on 8 SMT
siblings is slower *and* far less predictable, while 6 costs almost nothing.

### Where the 294 ms goes

A dictation of the 1 s benchmark clip costs **294 ms** of inference by the
built-in engine — median over 15 alternating A/B runs against the previous
build, which sits at 305 ms (IQR 282–306 vs 298–310). Single runs are not
evidence here: the same binary swings between 278 and 359 ms depending on what
else the laptop is doing. Inside the median run:

| stage | ms | share | arithmetic |
|---|---|---|---|
| ffn (both macaron halves, SiLU ~7 inside) | 141.2 | 51% | 2.8 GMAC |
| attention (`qkv` 48.4 + scores/softmax/AV/out 14.3) | 62.7 | 23% | 1.4 GMAC |
| conv module (`pw1` 25.2, `pw2` 13.1, GLU 1.7, depthwise 4.3) | 44.2 | 16% | 1.4 GMAC |
| pre_encode (detail below) | 16.2 | 6% | 0.2 GMAC |
| layer norms (5 pre-norms + final) | 5.0 | 2% | — |
| residual adds | 0.6 | — | — |
| **encoder** | **276.9** | | ~6 GMAC |
| mel front end + TDT decoder + dispatch | ~25 | | |

The three matmul families are 82% of the encoder and they all land at
**15–24 GMAC/s aggregate**, i.e. about **one MAC per cycle per thread**. That
uniformity across the int8 (`vpmaddubsw`) and f32 (`vfmadd`) paths is the whole
diagnosis: it is not the arithmetic width (both loops allow 8–10 MAC/cycle) and
it is not DRAM (the harness streams all 566 MB of `Q8_0` weights in ~180 ms =
**3.1 GB/s** against ~30 GB/s available, and `lin_task` already visits every
weight row once per utterance). It is per-block latency in the madd → widen →
convert → `fma` chain. That is what the next steps attack.

`pre_encode` alone, from `hex` stats (`PRE_STAGE` in the harness):

| stage | ms |
|---|---|
| conv.0 3×3 s2 1→256 (+relu) | 2.1 |
| conv.2 depthwise 3×3 s2 | 5.1 |
| conv.3 pointwise + relu | 3.2 |
| conv.5 depthwise 3×3 s2 | 1.4 |
| conv.6 pointwise + relu | 1.9 |
| `[c][t][f]` → `[t][c*f]` stack | 0.2 |
| `enc.pre_encode.out` Q8_0 4096→1024 | 2.4 |

The same breakdown on the previous build, where the pointwise stages ran as
im2col + transposed matmul + writeback: conv.0 **11.3 → 2.1**, conv.3 6.3 → 3.2,
conv.6 1.7 → 1.9 (unchanged), `out` **5.8 → 2.4**; the two depthwise stages are
untouched. Front end **32.8 → 16.6 ms**, medians over 5 runs each with
non-overlapping ranges.

Against the reference on the same clip:

| | infer (median) | model load | transcript |
|---|---|---|---|
| Odin engine | 294 ms | 866–1051 ms | byte-identical |
| `transcribe.cpp` / ggml | 313 ms | 1147–1174 ms | — |

So: at parity on compute within noise, and the model opens ~20% sooner (920 vs 1174 ms median) because
the encoder weights stay in their shipped `Q8_0`/`F16` form instead of being
decoded to f32. `hex engine shim` (or `HEX_ASR=shim`) forces the reference path
— the benchmark harness pins it that way — and `hex engine phonon` forces the
built-in engine. The env var wins over the config file; `hex status` prints the
resolved engine.

Process niceties around the kernels: the burst runs at `ABOVE_NORMAL_PRIORITY`
(337 → 306 ms median, interleaved) and both paths clear Windows EcoQoS before
inferring, since the idle process deliberately sits in it.

## Overhead

Measured with the model unloaded (a dictation itself is a separate burst):

| state | before | after |
|---|---|---|
| idle, HUD hidden | 1.1% of a core, 50 MB private | **0.2% of a core, 21 MB private** |
| recording (HUD visible) | 12.0% of a core | **5.0% of a core** |
| wakeups while idle | 60 Hz timer + 100 Hz pump poll | one 5 s housekeeping tick |

The HUD renders at 24 fps, its plate/rim/shadow are baked once so the per-frame
pass only does glow + dots, and the window timer is killed while hidden. The
capture pump starts with the hotkey and is joined on release, so idle costs no
polling. The ring is 2 minutes, not 10. Memory returns to baseline within a few
seconds of the idle release: a dictation costs ~400 MB for its duration and
nothing while you are not using it.

## Build

```powershell
# Odin app (~5 s): bin\hex.exe
.\build.ps1                 # debug
.\build.ps1 -Release

# Heavy half, built once: ggml + transcribe.cpp (CPU backend, no Vulkan SDK)
cmake -S native/third_party/transcribe.cpp -B native/build-transcribe -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF
cmake --build native/build-transcribe --config Release --parallel

# Shim, linked against those static libs, copied next to the exe in bin\
.\build.ps1 -Release -Native
```

`bin\hex.exe` runs without the shim — that is the built-in engine; `-Native`
only adds the reference/other-family path. Vulkan is a flag away
(`-DGGML_VULKAN=ON`) but needs the Vulkan SDK, which is not installed here; CPU
is the default. If `native/build-transcribe/.../transcribe.lib` is missing the
shim builds as a stub, so the app stays runnable.

`build.ps1` is location-agnostic: it resolves everything from `$PSScriptRoot`,
so it builds wherever the tree sits and from whatever directory you call it.
It finds the Odin compiler from `-Odin <path>`, then `$env:ODIN` (exe or
directory), then `PATH`, then the usual install directories (`<dir>\dist\odin.exe`,
as release archives unpack); `rc.exe` and `python` are discovered the same way
and either may be absent (an existing `bin\hex.res` is reused, or the exe is
built iconless). Everything it emits — the exe, its PDB, the icon intermediates
and the shim DLL — lands in `bin\`, which `.gitignore` covers. Any Odin dev
build from 2026-10 or later works; that is the one this was measured with.

## Benchmarks

Two harnesses, both one fresh process per measurement, both against the local
`%APPDATA%\hex\models\phonon-2.5-en-Q8_0.gguf`.

`bench-kernels/src` — `odin build bench-kernels/src -out:kern.exe -o:speed -microarch:znver2`:

| command | what it measures |
|---|---|
| `kern transcribe <gguf> <wav> [threads] [stats]` | end to end: `infer_ms`, tokens, transcript; `stats` adds the `ENC_STAGE` and `PRE_STAGE` split |
| `kern encoder <gguf> [frames] [threads]` | every 2-D matmul tensor in the model once at a fixed frame count, with aggregate GFLOP/s |
| `kern parity <gguf> <wav> <dump_dir>` | transcript against `transcribe.cpp`, byte for byte |
| `kern parity-enc <gguf> <dump_dir>` | the encoder stage by stage against reference dumps |
| `kern parcheck <gguf> [threads] [tensor]` | a pooled matmul against the single-threaded one, printing the first divergent row |

`autoresearch.sh` drives the same clip through both engines and prints
`infer_ms`, the `enc`/`dec`/`mel` split, `model_ms`, whole-process
`transcribe_ms`, the shim's `ref_infer_ms` and `parity` as medians over `RUNS`
runs, building into `.swarm/ar/build` so `bin\hex.exe` is never touched.

The per-stage split is `stats`-gated: `phonon.set_stats` hands the encoder a
pointer, every stage mark is a pair of `tick_now`, and with the pointer nil the
encoder times nothing at all.

## Models

Shared data root `%APPDATA%\hex\models`. `scripts/gen_models.py` filters Handy's
catalog down to two entries, so `hex models list` cannot drift back to a menu of
70 checkpoints; adding a model is data (`MODELS` in `src/models.odin`), not code.

**Phonon-2.5 English** (`phonon25_en`, 705 MB, default) is the one that cannot be
downloaded as-is: upstream ships a packed 164 MB container and the public GGUF
conversions use incompatible tensor names, so it is converted once:

```powershell
./scripts/fetch-phonon25.ps1 -Install
```

That pins and verifies the upstream artifact, expands it to a dense F32 GGUF
with transcribe.cpp's own converter (driven by `scripts/convert-phonon25.py`,
which reuses the upstream Fermion reader with the 2.5 hashes), quantises with
`transcribe-quantize`, and drops `phonon-2.5-en-Q8_0.gguf` into
`%APPDATA%\hex\models`. Python/torch appear only in that one-time conversion;
loading, dictation and the GUI are Odin + C++.

Switch with `hex models use <id>`; a config pointing at a missing model falls
back to the default instead of failing at startup. Phonon-2 was removed — same
0.6B class, and 2.5 no longer drops spaces (`by25 -30` became `by 25 to 30`).
Every other family (Canary) runs through the shim; the Odin engine implements the
Parakeet encoder and TDT decoder only.

Log lines from the C++ side are named after the architecture, not the file:
Phonon-2.5 is a Parakeet-family checkpoint, so `parakeet: weights memory-mapped`
is expected. The shim prints `[hex] model file: ...` right before it so the
actual GGUF is never in doubt.

## Layout

```
src/main.odin           CLI dispatch
src/capture.odin        miniaudio capture, 2-minute lock-free ring, level meter
src/wav.odin            RIFF writer, 16-bit reader with mono downmix + resample
src/transcribe.odin     shim loader, registry→path, engine dispatch, thread sizing
src/phonon_engine.odin  built-in ASR engine lifecycle (phonon/ package)
phonon/gguf.odin        GGUF reader: metadata, tensor table, Q8_0/F16/F32 views
phonon/mel.odin         NeMo log-mel frontend (matches ParakeetFeatureExtractor)
phonon/encoder.odin     subsampling convs + 24 conformer blocks (relative attention, conv module, macaron FFN)
phonon/decoder.odin     TDT decoder: predictor LSTM, joint, duration head
phonon/linalg.odin      Mat, tensor views, thread pool, parallel matmul/norm/silu
phonon/qkernels.odin    AVX2/FMA kernels for Q8_0, F16 and f32 weights
bench-kernels/          standalone kernel/engine timing driver
native/                 C ABI shim + CMakeLists (FetchContent at the pinned rev)
```

## Lineage

The maths is not invented here. The upstream Phonon reference (Rust, kept in
`phonon-rs-master/` for reading) defines the model; `transcribe.cpp`/`ggml` is
the executable authority on what the numbers should be, so every kernel and
stage was checked against it rather than against a paper. The Odin work is the
port plus the kernels: ggml's `Q8_0` mul_mat is the design the dot borrows
(activation quantised per frame, int8 weights dotted with `vpmaddubsw`, scale
folded per block), and `phonon/conv_check_test.odin` checks each conv2d
dispatch shape against the naive definition.

| Rust | Odin |
|---|---|
| `src/windows_audio.rs`, `cpal` | `capture.odin` (vendor:miniaudio) |
| `src/windows_paste.rs` | `paste.odin` |
| `src/transcription_models.rs` | `models.odin` (data only) |
| `src/gguf_session.rs` | `phonon/gguf.odin` + `phonon/*.odin` (engine), `native/transcribe_shim.cpp` (reference) |
| `src/rune_hud.rs` + `rune_hud.wgsl` | `hud.odin` (DIB) |
| `src/rune_ui/*` (56k lines) | not ported — window/pane system |
| `src/local_api.rs` | not ported |
| `src/updater.rs`, release protocol | not ported |

## Next steps, in order

1. **Four frames per weight pass.** The two-frame kernel already proves weight
   traffic is the limiter (35 GB/s at a 63 GFLOPS peak); four frames per block
   load is the next doubling, with the same accumulator-tree discipline so the
   output stays bit-identical.
2. **Keep the f16 tensors as f16** — today the 48 of them are promoted at load;
   an F16C dot over the shipped halves removes the promotion pass and the extra copy.
3. Then the app surface that is still missing: `rune_ui` panes, `local_api`,
   updater.
