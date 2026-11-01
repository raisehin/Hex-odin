#!/usr/bin/env bash
# Benchmark: end-to-end `hex transcribe` of the 1 s clip, once per fresh process.
# Input is "one second.mp3" decoded to 16 kHz mono s16 (ffmpeg is deterministic),
# because wav_read_mono16k only accepts RIFF/PCM.
#
# Metrics
#   transcribe_ms   median wall time of N runs (lower is better)
#   same_text       1 when every run produced the recorded transcript
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

mkdir -p "$BUILD"

# 1. Input audio (cached, rebuilt when the mp3 changes).
if [ ! -f "$WAV" ] || [ "$SRC" -nt "$WAV" ]; then
	ffmpeg -y -loglevel error -i "$SRC" -ac 1 -ar 16000 -c:a pcm_s16le "$WAV" || exit 1
fi

# 2. Build the app from the live tree.
rm -f "$BUILD"/*.odin
cp "$ROOT"/src/*.odin "$BUILD"/ || exit 1
"$ODIN" build "$BUILD" -out:"$BUILD/hex.exe" -o:speed || exit 1

# 3. Copy the shim next to the test binary so dynlib finds it.
if [ -f "$ROOT/native/bin/transcribe_shim.dll" ]; then
	cp "$ROOT/native/bin/transcribe_shim.dll" "$BUILD/transcribe_shim.dll"
elif [ -f "$ROOT/transcribe_shim.dll" ]; then
	cp "$ROOT/transcribe_shim.dll" "$BUILD/transcribe_shim.dll"
fi

# 4. Time N runs, print metrics.
HEX_BIN="$BUILD/hex.exe" HEX_WAV="$WAV" HEX_RUNS="$RUNS" python - <<'PY'
import hashlib, os, statistics, subprocess, sys, time

exe, wav, runs = os.environ["HEX_BIN"], os.environ["HEX_WAV"], int(os.environ["HEX_RUNS"])
times, texts, errs = [], [], []
for _ in range(runs):
    t0 = time.perf_counter()
    p = subprocess.run([exe, "transcribe", wav], capture_output=True, text=True)
    times.append((time.perf_counter() - t0) * 1000.0)
    texts.append(p.stdout.strip())
    errs.append(p.stderr)

if any(t is None or len(t) == 0 for t in texts):
    print("ASI error=empty_transcript")
    print("METRIC same_text=0")
    sys.exit(1)

digests = sorted({hashlib.sha256(t.encode()).hexdigest()[:16] for t in texts})
median = statistics.median(times)
print("ASI times_ms=" + ",".join(f"{t:.0f}" for t in times))
print("ASI transcript=" + texts[0].replace("\n", " ")[:200])
print("ASI transcript_sha=" + "|".join(digests))
print("ASI stderr_tail=" + errs[-1].strip().splitlines()[-1][:160] if errs[-1].strip() else "ASI stderr_tail=")
print(f"METRIC transcribe_ms={median:.1f}")
print(f"METRIC same_text={1 if len(digests) == 1 else 0}")
PY
