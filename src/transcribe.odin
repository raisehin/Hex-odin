package main

import "core:dynlib"
import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"

// Shim ABI (native/transcribe_shim.h). The shim wraps transcribe.cpp/ggml and
// is loaded at runtime, so hex.exe keeps working before the C++ side is built.
Shim :: struct {
	lib:        dynlib.Library,
	version:    proc "c" () -> cstring,
	init:       proc "c" (model: cstring) -> i32,
	transcribe: proc "c" (samples: [^]f32, n: i32, language: cstring) -> cstring,
	free_text:  proc "c" (p: cstring),
	list_devices: proc "c" () -> cstring,
	free_devices: proc "c" (p: cstring),
	vad_init:    proc "c" (model: cstring) -> i32,
	vad_speech:  proc "c" (samples: [^]f32, n: i32) -> i32,
	vad_free:    proc "c" (),
	set_threads: proc "c" (n: i32),
	close:       proc "c" (),
	last_error:  proc "c" () -> cstring,
}

sym :: proc(lib: dynlib.Library, name: string) -> rawptr {
	p, ok := dynlib.symbol_address(lib, name)
	return ok ? p : nil
}

shim: Shim
shim_tried: bool

transcribe_available :: proc() -> bool {
	if shim_tried { return shim.lib != nil }
	shim_tried = true
	exe_dir, _ := os.get_executable_directory(context.temp_allocator)
	candidates := []string{
		strings.concatenate({exe_dir, "/transcribe_shim.dll"}, context.temp_allocator),
		strings.concatenate({exe_dir, "/native/bin/transcribe_shim.dll"}, context.temp_allocator),
		strings.concatenate({exe_dir, "/native/build/Release/transcribe_shim.dll"}, context.temp_allocator),
		"transcribe_shim.dll",
	}
	for name in candidates {

		lib, ok := dynlib.load_library(name)
		if !ok { continue }
		shim = Shim{lib = lib}
		shim.version = cast(proc "c" () -> cstring)sym(lib, "hex_shim_version")
		shim.init = cast(proc "c" (cstring) -> i32)sym(lib, "hex_model_init")
		shim.transcribe = cast(proc "c" ([^]f32, i32, cstring) -> cstring)sym(lib, "hex_transcribe")
		shim.free_text = cast(proc "c" (cstring))sym(lib, "hex_free")
		shim.list_devices = cast(proc "c" () -> cstring)sym(lib, "hex_list_devices")
		shim.free_devices = cast(proc "c" (cstring))sym(lib, "hex_free")
		shim.vad_init = cast(proc "c" (cstring) -> i32)sym(lib, "hex_vad_init")
		shim.vad_speech = cast(proc "c" ([^]f32, i32) -> i32)sym(lib, "hex_vad_has_speech")
		shim.vad_free = cast(proc "c" ())sym(lib, "hex_vad_free")
		shim.set_threads = cast(proc "c" (i32))sym(lib, "hex_set_threads")
		shim.close = cast(proc "c" ())sym(lib, "hex_model_close")
		shim.last_error = cast(proc "c" () -> cstring)sym(lib, "hex_last_error")
		if shim.init == nil || shim.transcribe == nil {
			dynlib.unload_library(lib)
			shim = {}
			continue
		}
		return true
	}
	return false
}

shim_error_text :: proc() -> string {
	if shim.last_error == nil { return "" }
	p := shim.last_error()
	if p == nil { return "" }
	return strings.clone(string(p), context.temp_allocator)
}

transcribe_devices :: proc() -> []string {
	out: [dynamic]string
	if !transcribe_available() || shim.list_devices == nil { return out[:] }
	s := shim.list_devices()
	defer shim.free_devices(s)
	text := string(s)
	for line in strings.split_lines(text) {
		if len(line) > 0 { append(&out, strings.clone(line)) }
	}
	return out[:]
}

// The model is ~700 MB of mmap'd weights. Keep it for 3 minutes after the last
// dictation so consecutive uses stay instant, then hand the memory back; the
// next run reopens it in ~1 s. HEX_MODEL_IDLE_SECS overrides the timeout.
MODEL_IDLE_SECONDS :: 180

model_lock: sync.Mutex
model_live: bool
live_backend: Asr_Backend
model_last: time.Time

model_idle_seconds :: proc() -> f64 {
	if s := os.get_env("HEX_MODEL_IDLE_SECS", context.temp_allocator); len(s) > 0 {
		if n, ok := strconv.parse_int(s); ok && n > 0 { return f64(n) }
	}
	return MODEL_IDLE_SECONDS
}

model_idle_worker :: proc(_: ^thread.Thread) {
	limit := model_idle_seconds()
	for {
		time.sleep(5 * time.Second)
		// Safety net for batched log lines written outside a dictation.
		log_flush()
		sync.lock(&model_lock)
		if model_live && time.duration_seconds(time.since(model_last)) >= limit {
			model_close_locked()
			sync.unlock(&model_lock)
			log_write("[hex] model released after %d s idle", i32(limit))
			continue
		}
		sync.unlock(&model_lock)
	}
}

// ggml's thread pool spins between graph nodes, so handing it every logical CPU
// on an SMT part oversubscribes the physical cores: measured on a 4C/8T Zen2,
// 8 threads ran the 1 s clip at 459 ms median (up to 1.7 s) versus 353 ms at 6,
// and one core at 8 threads costs 2-3x. Three quarters of the logical count
// leaves one SMT sibling free per physical core; 0 keeps meaning "auto".
worker_threads :: proc(want: int) -> int {
	if want > 0 { return want }
	n := logical_cpu_count()
	if n <= 2 { return n }
	return n - n / 4
}


// Init is 890 ms cold (measured) and the idle worker hands the session back
// after MODEL_IDLE_SECONDS, so the wait used to land between the key release
// and the paste. Preloading on the key down overlaps it with the recording.
preload_running: i32

model_close_locked :: proc() {
	if !model_live { return }
	if live_backend == .Phonon {
		phonon_close()
	} else if shim.close != nil {
		shim.close()
	}
	model_live = false
}

model_ensure_locked :: proc(cfg: ^Config, model_id: string, backend: Asr_Backend) -> (ok: bool, err: string) {
	if model_live && live_backend == backend { return true, "" }
	// Switching model family (shim <-> built-in engine) drops the old session
	// first: both hold the whole mapping, and only one is ever used at a time.
	model_close_locked()
	path := model_path(cfg, model_id)
	if path == "" { return false, fmt.aprintf("unknown model: %s", model_id) }
	if !file_exists(path) { return false, fmt.aprintf("model not installed: %s", path) }
	switch backend {
	case .Phonon:
		// The built-in engine needs no thread hint: its pool has none of ggml's
		// per-node barriers, so `threads = 0` there means the full logical count
		// and SMT siblings are worth using (interleaved 6 vs 8 on the 1 s clip:
		// 339 vs 334 ms median, and 6 spun up 589-599 ms outliers).
		if ok, e := phonon_open(path, cfg.threads); !ok { return false, e }
	case .Shim:
		if shim.set_threads != nil { shim.set_threads(i32(worker_threads(cfg.threads))) }
		init := shim.init(strings.clone_to_cstring(path, context.temp_allocator))
		if init != 0 {
			if detail := shim_error_text(); len(detail) > 0 {
				return false, fmt.aprintf("model init failed (%d): %s", init, detail)
			}
			return false, fmt.aprintf("model init failed (%d)", init)
		}
	}
	model_live = true
	live_backend = backend
	model_last = time.now()
	if m := model_by_id(model_id); m != nil {
		log_write("[hex] model %s (%s) on %s", model_id, m.filename, engine_name(backend))
	}
	return true, ""
}

preload_worker :: proc(_: ^thread.Thread) {
	cfg := config_load()
	// The VAD model is 7.5 ms and belongs to the first dictation only.
	_ = vad_load()
	sync.lock(&model_lock)
	_, _ = model_ensure_locked(&cfg, cfg.model, asr_backend_for(cfg.model))
	sync.unlock(&model_lock)
	sync.atomic_store(&preload_running, i32(0))
}

// Warms the VAD model and the ASR session while the user is still speaking.
// Never blocks the caller and never races the transcription: the worker takes
// the same model lock, so a fast key release just waits for the preload.
preload_model_async :: proc() {
	cfg := config_load()
	if asr_backend_for(cfg.model) == .Shim && !transcribe_available() { return }
	if sync.atomic_load(&preload_running) != 0 { return }
	sync.atomic_store(&preload_running, i32(1))
	if t := thread.create(preload_worker); t != nil {
		t.flags += {.Self_Cleanup}
		thread.start(t)
	} else {
		sync.atomic_store(&preload_running, i32(0))
	}
}

// A dictation ends where the speech ends. The room tone after the last word is
// fed to the encoder as real frames, and encoder frames are what a dictation
// costs: on the 1.056 s benchmark clip (466 ms of trailing silence) the trim
// takes inference from 322 ms to ~255 ms. A pad is kept after the last speech
// frame so the acoustic model still sees an endpoint, and clips whose tail is
// shorter than TRIM_MIN_CUT_MS are sent untouched.
TRIM_TAIL_PAD_MS :: 250
TRIM_MIN_CUT_MS :: 100
TRIM_FRAME_MS :: 20
// A frame is speech when its mean square is above this fraction of the loudest
// frame (-30 dB), and only runs of at least TRIM_RUN_FRAMES count: a keyboard
// click or a mouse button 300 ms after the last syllable is 40 ms of wideband
// noise and must not hold 200 ms of trailing silence alive.
TRIM_REL_FLOOR :: 0.001
TRIM_RUN_FRAMES :: 5

trim_trailing_silence :: proc(samples: []f32) -> (out: []f32, trimmed_ms: int) {
	out = samples
	rate := 16_000
	frame := rate * TRIM_FRAME_MS / 1000
	if len(samples) < frame * 8 { return }
	peak_ms: f32 = 0
	frames := (len(samples) + frame - 1) / frame
	energy := make([]f32, frames, context.temp_allocator) or_else nil
	if energy == nil { return }
	for i in 0 ..< frames {
		off := i * frame
		acc: f32 = 0
		for s in samples[off:min(off + frame, len(samples))] { acc += s * s }
		energy[i] = acc / f32(frame)
		if energy[i] > peak_ms { peak_ms = energy[i] }
	}
	if peak_ms < 1e-8 { return }
	limit := peak_ms * f32(TRIM_REL_FLOOR)
	last := 0
	run := 0
	for i in 0 ..< frames {
		if energy[i] > limit {
			if run += 1; run >= TRIM_RUN_FRAMES { last = (i + 1) * frame }
		} else {
			run = 0
		}
	}
	keep := min(last + rate * TRIM_TAIL_PAD_MS / 1000, len(samples))
	cut := len(samples) - keep
	if cut < rate * TRIM_MIN_CUT_MS / 1000 { return }
	trimmed_ms = cut * 1000 / rate
	return samples[:keep], trimmed_ms
}

transcribe_run :: proc(model_id: string, samples: []f32, language := "") -> (text: string, err: string) {
	cfg := config_load()
	backend := asr_backend_for(model_id)
	// The built-in engine needs no DLL; only the shim path does.
	if backend == .Shim && !transcribe_available() {
		return "", strings.clone("transcribe shim not built (scripts/build-native.ps1)")
	}

	sync.lock(&model_lock)
	defer sync.unlock(&model_lock)

	t_model := time.tick_now()
	if ok, e := model_ensure_locked(&cfg, model_id, backend); !ok { return "", e }
	model_last = time.now()

	samples, trimmed_ms := trim_trailing_silence(samples)
	use_lang := language if len(language) > 0 else cfg.language
	t_asr := time.tick_now()
	// Measured here, before inference starts, so this stays a pure model wait.
	// Taking it after inference made it "everything elapsed" and the two fields
	// printed identical numbers (3162/3162, 704/703).
	wait_ms := time.duration_milliseconds(time.tick_since(t_model))
	set_transcribe_priority(true)
	res_text: string
	res_err: string
	if backend == .Phonon {
		res_text, res_err = phonon_transcribe(samples)
	} else {
		lang := strings.clone_to_cstring(use_lang, context.temp_allocator)
		res := shim.transcribe(raw_data(samples), i32(len(samples)), lang)
		if res == nil {
			res_err = strings.clone("transcribe returned no text")
		} else {
			res_text = strings.clone(string(res))
			shim.free_text(res)
		}
	}
	set_transcribe_priority(false)
	// One line per dictation: the two stages a user waits on. Model wait is 0
	// when the preload already warmed the session, and the whole cost after an
	// idle release, so the log answers "why was that one slow" directly.
	if res_err == "" {
		infer_ms := time.duration_milliseconds(time.tick_since(t_asr))
		log_write("[hex] asr: model-wait %.0f ms, inference %.0f ms (%d samples, %d ms tail trimmed)",
			wait_ms, infer_ms, len(samples), trimmed_ms)
		// Same numbers on stderr for the benchmark harness (autoresearch.sh),
		// which cannot read the log file reliably. Off unless asked for.
		if len(os.get_env("HEX_PROFILE", context.temp_allocator)) > 0 {
			fmt.eprintf("PROFILE model_ms=%.0f infer_ms=%.0f samples=%d\n", wait_ms, infer_ms, len(samples))
		}
	}
	return res_text, res_err
}


// VAD (Silero) is optional: load it when the model file exists and let the
// caller fall back to a level check when it does not.
VAD_FILE :: "silero-vad-v6.2-F32.gguf"
vad_ready: bool
vad_tried: bool

vad_load :: proc() -> bool {
	if vad_tried { return vad_ready }
	vad_tried = true
	if !transcribe_available() || shim.vad_init == nil { return false }
	cfg := config_load()
	path := strings.concatenate({cfg.models_dir, "/", VAD_FILE}, context.temp_allocator)
	if !file_exists(path) { return false }
	rc := shim.vad_init(strings.clone_to_cstring(path, context.temp_allocator))
	if rc != 0 {
		if detail := shim_error_text(); len(detail) > 0 {
			log_write("vad load failed (%d): %s", rc, detail)
		} else {
			log_write("vad load failed (%d)", rc)
		}
		return false
	}
	vad_ready = true
	return true
}

// true when the clip contains speech. Silence (or a missing VAD) returns true
// so a missing model never blocks dictation.
vad_has_speech :: proc(samples: []f32) -> bool {
	if !vad_ready || shim.vad_speech == nil || len(samples) == 0 { return true }
	return shim.vad_speech(raw_data(samples), i32(len(samples))) == 1
}
