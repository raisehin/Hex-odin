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
			if shim.close != nil { shim.close() }
			model_live = false
			sync.unlock(&model_lock)
			log_write("[hex] model released after %d s idle", i32(limit))
			continue
		}
		sync.unlock(&model_lock)
	}
}

// Init is 890 ms cold (measured) and the idle worker hands the session back
// after MODEL_IDLE_SECONDS, so the wait used to land between the key release
// and the paste. Preloading on the key down overlaps it with the recording.
preload_running: i32

model_ensure_locked :: proc(cfg: ^Config, model_id: string) -> (ok: bool, err: string) {
	if model_live { return true, "" }
	path := model_path(cfg, model_id)
	if path == "" { return false, fmt.aprintf("unknown model: %s", model_id) }
	if !file_exists(path) { return false, fmt.aprintf("model not installed: %s", path) }
	if shim.set_threads != nil { shim.set_threads(i32(cfg.threads)) }
	init := shim.init(strings.clone_to_cstring(path, context.temp_allocator))
	if init != 0 {
		if detail := shim_error_text(); len(detail) > 0 {
			return false, fmt.aprintf("model init failed (%d): %s", init, detail)
		}
		return false, fmt.aprintf("model init failed (%d)", init)
	}
	model_live = true
	model_last = time.now()
	if m := model_by_id(model_id); m != nil { log_write("[hex] model %s (%s)", model_id, m.filename) }
	return true, ""
}

preload_worker :: proc(_: ^thread.Thread) {
	cfg := config_load()
	// The VAD model is 7.5 ms and belongs to the first dictation only.
	_ = vad_load()
	sync.lock(&model_lock)
	_, _ = model_ensure_locked(&cfg, cfg.model)
	sync.unlock(&model_lock)
	sync.atomic_store(&preload_running, i32(0))
}

// Warms the VAD model and the ASR session while the user is still speaking.
// Never blocks the caller and never races the transcription: the worker takes
// the same model lock, so a fast key release just waits for the preload.
preload_model_async :: proc() {
	if !transcribe_available() { return }
	if sync.atomic_load(&preload_running) != 0 { return }
	sync.atomic_store(&preload_running, i32(1))
	if t := thread.create(preload_worker); t != nil {
		t.flags += {.Self_Cleanup}
		thread.start(t)
	} else {
		sync.atomic_store(&preload_running, i32(0))
	}
}

transcribe_run :: proc(model_id: string, samples: []f32, language := "") -> (text: string, err: string) {
	if !transcribe_available() {
		return "", strings.clone("transcribe shim not built (scripts/build-native.ps1)")
	}
	cfg := config_load()

	sync.lock(&model_lock)
	defer sync.unlock(&model_lock)

	t_model := time.tick_now()
	if ok, e := model_ensure_locked(&cfg, model_id); !ok { return "", e }
	model_last = time.now()

	use_lang := language if len(language) > 0 else cfg.language
	lang := strings.clone_to_cstring(use_lang, context.temp_allocator)
	t_asr := time.tick_now()
	res := shim.transcribe(raw_data(samples), i32(len(samples)), lang)
	// One line per dictation: the two stages a user waits on. Model wait is 0
	// when the preload already warmed the session, and the whole cost after an
	// idle release, so the log answers "why was that one slow" directly.
	if res != nil {
		wait_ms := time.duration_milliseconds(time.tick_since(t_model))
		infer_ms := time.duration_milliseconds(time.tick_since(t_asr))
		log_write("[hex] asr: model-wait %.0f ms, inference %.0f ms (%d samples)",
			wait_ms, infer_ms, len(samples))
		// Same numbers on stderr for the benchmark harness (autoresearch.sh),
		// which cannot read the log file reliably. Off unless asked for.
		if len(os.get_env("HEX_PROFILE", context.temp_allocator)) > 0 {
			fmt.eprintf("PROFILE model_ms=%.0f infer_ms=%.0f samples=%d\n", wait_ms, infer_ms, len(samples))
		}
	}
	if res == nil { return "", strings.clone("transcribe returned no text") }
	defer shim.free_text(res)
	return strings.clone(string(res)), ""
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
