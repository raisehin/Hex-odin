#!/usr/bin/env bash
# Benchmark: the Odin engine (phonon/) transcribing one fixed clip, one fresh
# process per run, timed against the transcribe.cpp shim on the same input.
#
# Input is "bench-kernels/Hello there.wav" (16 kHz mono s16), fixed on disk, so
# every run does exactly the same work. Model is the local Phonon-2.5 Q8_0
# GGUF; nothing touches the network.
#
# Metrics (medians over RUNS runs unless noted; lower is better except parity):
#   infer_ms    Odin engine ASR compute (primary)
#   enc_ms / dec_ms / mel_ms   stage split of infer_ms
#   model_ms    Odin engine model load before inference
#   transcribe_ms   whole-process wall time (spawn + load + infer + teardown)
#   ref_infer_ms    transcribe.cpp/ggml doing the same clip (the baseline to beat)
#   parity      1 when the Odin transcript matches the shim's byte for byte
#
# Build output goes to .swarm/ar/build so ./hex.exe is never touched.
set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
AR="$ROOT/.swarm/ar"
BUILD="$AR/build"
ODIN="C:/Users/rhais/Documents/odin-windows-amd64-dev-2026-10/dist/odin.exe"
WAV="$ROOT/bench-kernels/Hello there.wav"
MODEL="${HEX_BENCH_MODEL:-$APPDATA/hex/models/phonon-2.5-en-Q8_0.gguf}"
RUNS="${HEX_BENCH_RUNS:-5}"
REF_RUNS="${HEX_BENCH_REF_RUNS:-3}"

mkdir -p "$BUILD" || exit 1
if [ ! -f "$WAV" ]; then echo "autoresearch: missing $WAV" >&2; exit 1; fi
if [ ! -f "$MODEL" ]; then echo "autoresearch: missing model $MODEL" >&2; exit 1; fi

# 1. Odin engine driver (bench-kernels/src -> kern.exe).
KERN="$BUILD/kern.exe"
newer_engine=$(find "$ROOT/bench-kernels/src" "$ROOT/phonon" -name '*.odin' -newer "$KERN" 2>/dev/null | head -1)
if [ ! -f "$KERN" ] || [ -n "$newer_engine" ]; then
  "$ODIN" build "$ROOT/bench-kernels/src" -out:"$KERN" -o:speed -microarch:znver2 || {
    echo "autoresearch: odin build (engine) failed" >&2; exit 1; }
fi

# 2. Reference driver (src/ -> hex.exe) for the baseline number. Needs the shim
#    DLL next to it; the repo root copy is the built one. The reference runs are
#    pinned to that DLL with HEX_ASR=shim (see step 4).
HEX="$BUILD/hex.exe"
newer_app=$(find "$ROOT/src" -name '*.odin' -newer "$HEX" 2>/dev/null | head -1)
if [ ! -f "$HEX" ] || [ -n "$newer_app" ]; then
  "$ODIN" build "$ROOT/src" -out:"$HEX" -o:speed || {
    echo "autoresearch: odin build (hex) failed" >&2; exit 1; }
fi
if [ -f "$ROOT/transcribe_shim.dll" ] && { [ ! -f "$BUILD/transcribe_shim.dll" ] || [ "$ROOT/transcribe_shim.dll" -nt "$BUILD/transcribe_shim.dll" ]; }; then
  cp "$ROOT/transcribe_shim.dll" "$BUILD/transcribe_shim.dll" || exit 1
fi

# 3. Warm-up (page the 705 MB mapping in once); its numbers are discarded.
warm=$("$KERN" transcribe "$MODEL" "$WAV" 2>&1)
printf '%s\n' "$warm" | grep -q '^PROFILE ' || {
  echo "autoresearch: engine warm-up failed" >&2
  printf '%s\n' "$warm" >&2; exit 1; }

infer_list=""; model_list=""; wall_list=""
mel_list=""; enc_list=""; dec_list=""
texts=""

for i in $(seq 1 "$RUNS"); do
  start=$(date +%s%N)
  # The driver may exit nonzero after printing (decoder is still WIP); the
  # PROFILE line is the contract, so check for it instead of the exit code.
  out=$("$KERN" transcribe "$MODEL" "$WAV" 2>&1)
  end=$(date +%s%N)

  profile=$(printf '%s\n' "$out" | grep -m1 '^PROFILE ')
  if [ -z "$profile" ]; then
    echo "autoresearch: engine run $i printed no PROFILE line" >&2
    printf '%s\n' "$out" >&2; exit 1
  fi
  m=$(printf '%s\n' "$profile" | sed -n 's/.*model_ms=\([0-9]*\).*/\1/p')
  f=$(printf '%s\n' "$profile" | sed -n 's/.*infer_ms=\([0-9]*\).*/\1/p')
  st=$(printf '%s\n' "$out" | grep -m1 '^PHONON_STAGES ')
  mel=$(printf '%s\n' "$st" | sed -n 's/.*mel=\([0-9.]*\).*/\1/p')
  enc=$(printf '%s\n' "$st" | sed -n 's/.*enc=\([0-9.]*\).*/\1/p')
  dec=$(printf '%s\n' "$st" | sed -n 's/.*dec=\([0-9.]*\).*/\1/p')
  txt=$(printf '%s\n' "$out" | grep -v '^PROFILE\|^PHONON_STAGES\|^DBG\|^  \[\|^\[hex\]\|^parakeet\|^decoder:' | grep -v '^[[:space:]]*$' | tail -1)

  texts="$texts$txt
"
  infer_list="$infer_list$f
"; model_list="$model_list$m
"
  wall_list="$wall_list$(( (end - start) / 1000000 ))
"
  mel_list="$mel_list$mel
"; enc_list="$enc_list$enc
"; dec_list="$dec_list$dec
"
done

# 4. Reference baseline: same clip through transcribe.cpp/ggml.
ref_list=""
for i in $(seq 1 "$REF_RUNS"); do
  # HEX_ASR=shim keeps the reference on transcribe.cpp/ggml: hex.exe also
  # carries the Odin engine now, which would otherwise measure itself.
  rout=$(HEX_ASR=shim HEX_PROFILE=1 "$HEX" transcribe "$WAV" 2>&1)
  r=$(printf '%s\n' "$rout" | sed -n 's/.*infer_ms=\([0-9]*\).*/\1/p' | head -1)
  ref_list="$ref_list$r
"
  if [ "$i" = "1" ]; then
    ref_text=$(printf '%s\n' "$rout" | grep -v '^PROFILE\|^\[hex\]\|^parakeet\|^decoder:\|^ggml' | grep -v '^[[:space:]]*$' | tail -1)
  fi
done

median() { printf '%s\n' "$1" | grep -v '^$' | sort -n | awk '{a[NR]=$1} END { if (NR==0) {print 0} else if (NR%2) print a[(NR+1)/2]; else print int((a[NR/2]+a[NR/2+1])/2) }'; }

ours_text=$(printf '%s' "$texts" | grep -v '^$' | head -1)
parity=0
if [ "$ours_text" = "$ref_text" ]; then parity=1; fi

echo "METRIC infer_ms=$(median "$infer_list")"
echo "METRIC enc_ms=$(median "$enc_list")"
echo "METRIC dec_ms=$(median "$dec_list")"
echo "METRIC mel_ms=$(median "$mel_list")"
echo "METRIC model_ms=$(median "$model_list")"
echo "METRIC transcribe_ms=$(median "$wall_list")"
echo "METRIC ref_infer_ms=$(median "$ref_list")"
echo "METRIC parity=$parity"
echo "ours: $ours_text"
echo "ref:  $ref_text"
exit 0
