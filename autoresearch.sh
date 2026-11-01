#!/usr/bin/env bash
# Benchmark: `hex transcribe` of the fixed 1 s clip, one fresh process per run.
# Input is "one second.mp3" decoded once to 16 kHz mono s16 (ffmpeg is
# deterministic); wav_read_mono16k only accepts RIFF/PCM, so the mp3 cannot be
# fed directly.
#
# Metrics (medians over HEX_BENCH_RUNS runs, default 5):
#   infer_ms       CPU ASR compute inside the shim (primary; lower is better)
#   model_ms       model load/dequant wait before inference
#   transcribe_ms  whole-process wall time (spawn + load + infer + teardown)
#   same_text      1 when every run produced the same transcript
# Machine is CPU-only by construction: ggml runs with the CPU backend and the
# shim is built with GGML_VULKAN=OFF, no CUDA.
#
# Build goes to .swarm/ar/build (never touches ./hex.exe).
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
AR="$ROOT/.swarm/ar"
BUILD="$AR/build"
ODIN="C:/Users/rhais/Documents/odin-windows-amd64-dev-2026-10/dist/odin.exe"
SRC="$ROOT/one second.mp3"
WAV="$AR/in.wav"
RUNS="${HEX_BENCH_RUNS:-5}"
CONFIG="$APPDATA/hex/odin-config"

mkdir -p "$BUILD"

# 1. Input audio (cached, rebuilt when the mp3 changes).
if [ ! -f "$WAV" ] || [ "$SRC" -nt "$WAV" ]; then
	ffmpeg -y -loglevel error -i "$SRC" -ac 1 -ar 16000 -c:a pcm_s16le "$WAV" || exit 1
fi

# 2. Build the app from the live tree.
rm -f "$BUILD"/*.odin
cp "$ROOT"/src/*.odin "$BUILD"/ || exit 1
"$ODIN" build "$BUILD" -out:"$BUILD/hex.exe" -o:speed || exit 1

# 3. Shim next to the test binary (dynlib looks beside the exe first).
SHIM=""
for cand in "$ROOT/native/bin/transcribe_shim.dll" "$ROOT/native/build/Release/transcribe_shim.dll" "$ROOT/transcribe_shim.dll"; do
	if [ -f "$cand" ]; then SHIM="$cand"; break; fi
done
[ -n "$SHIM" ] || { echo "autoresearch: transcribe_shim.dll not found; run scripts/build-native.ps1" >&2; exit 1; }
cp "$SHIM" "$BUILD/transcribe_shim.dll"

# 4. Machine facts, recorded so every number is traceable to this box.
FACTS=$(powershell -NoProfile -Command '(Get-CimInstance Win32_Processor).Name + "|" + (Get-CimInstance Win32_VideoController | Select-Object -First 1).Name' 2>/dev/null | tr -d '\r')
export HEX_BENCH_CPU="${FACTS%%|*}"
export HEX_BENCH_GPU="${FACTS##*|}"
export HEX_BENCH_THREADS="$(sed -n 's/^threads=//p' "$CONFIG" 2>/dev/null)"
export HEX_BENCH_MODEL="$(sed -n 's/^model=//p' "$CONFIG" 2>/dev/null)"
export HEX_BENCH_BUNDLE="$(basename "$SHIM")"

# 5. Time N runs, print metrics. HEX_PROFILE makes the app print its internal
#    stage split on stderr; parsing it is what separates the ASR compute from
#    process startup and teardown.
HEX_BIN="$BUILD/hex.exe" HEX_WAV="$WAV" HEX_RUNS="$RUNS" HEX_PROFILE=1 python - <<'PY'
import hashlib, os, re, statistics, subprocess, sys, time

exe, wav, runs = os.environ["HEX_BIN"], os.environ["HEX_WAV"], int(os.environ["HEX_RUNS"])
prof = re.compile(r"PROFILE model_ms=(\d+) infer_ms=(\d+) samples=(\d+)")

walls, infers, models, texts, samples = [], [], [], [], []
for _ in range(runs):
    t0 = time.perf_counter()
    p = subprocess.run([exe, "transcribe", wav], capture_output=True, text=True)
    walls.append((time.perf_counter() - t0) * 1000.0)
    m = prof.search(p.stderr)
    if not m:
        print("ASI error=no_profile_line")
        print("ASI stderr_tail=" + p.stderr.strip().replace("\n", " ")[-200:])
        print("METRIC same_text=0")
        sys.exit(1)
    models.append(float(m.group(1)))
    infers.append(float(m.group(2)))
    samples.append(int(m.group(3)))
    texts.append(p.stdout.strip())

if any(not t for t in texts):
    print("ASI error=empty_transcript")
    print("METRIC same_text=0")
    sys.exit(1)

digests = sorted({hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts})
# 16 kHz s16 mono after a 44-byte canonical header; fall back to sample count.
audio_ms = samples[0] / 16.0
print("ASI cpu=" + os.environ.get("HEX_BENCH_CPU", "unknown"))
print("ASI gpu=" + os.environ.get("HEX_BENCH_GPU", "unknown"))
print("ASI bundle=" + os.environ.get("HEX_BENCH_BUNDLE", "unknown"))
print("ASI threads=" + os.environ.get("HEX_BENCH_THREADS", "unknown"))
print("ASI model=" + os.environ.get("HEX_BENCH_MODEL", "unknown"))
print("ASI audio_ms=%.1f" % audio_ms)
print("ASI walls_ms=" + ",".join("%.0f" % w for w in walls))
print("ASI infer_ms=" + ",".join("%.0f" % i for i in infers))
print("ASI model_load_ms=" + ",".join("%.0f" % i for i in models))
print("ASI rtf=%.3f" % (statistics.median(infers) / audio_ms))
print("ASI transcript=" + texts[0].replace("\n", " ")[:200])
print("ASI transcript_sha=" + "|".join(digests))
print("METRIC infer_ms=%.1f" % statistics.median(infers))
print("METRIC model_ms=%.1f" % statistics.median(models))
print("METRIC transcribe_ms=%.1f" % statistics.median(walls))
print("METRIC same_text=%d" % (1 if len(digests) == 1 else 0))
sys.exit(0 if len(digests) == 1 else 1)
PY
rc=$?

exit $rc
