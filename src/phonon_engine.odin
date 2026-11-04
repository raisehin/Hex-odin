package main

import "core:fmt"
import "core:os"
import "core:strings"
import "../phonon"

// asr_backend_for picks the engine for a model: the phonon port implements the
// parakeet encoder and the TDT decoder, so the phonon family runs on the Odin
// kernels in phonon/ (the ones bench-kernels/src times) with no DLL at all.
// Everything else, and the VAD, still goes through transcribe.cpp.
//
// HEX_ASR=shim|phonon|odin forces one engine; the benchmark harness pins shim
// so its reference number stays the C++ path.
Asr_Backend :: enum {
	Shim,
	Phonon,
}

asr_backend_for :: proc(model_id: string) -> Asr_Backend {
	if pref := os.get_env("HEX_ASR", context.temp_allocator); len(pref) > 0 {
		if b, ok := asr_backend_parse(pref); ok { return b }
	}
	if b, ok := asr_backend_parse(config_load().engine); ok { return b }
	if m := model_by_id(model_id); m != nil && m.family == "parakeet" {
		// The shim now links ggml's Vulkan backend, which runs this model's
		// encoder and decoder on the iGPU. Measured on a 10.9 s clip: 967 ms
		// through ggml-Vulkan against 1590 ms for the Odin kernels' encoder
		// alone, with identical text. Prefer it whenever the DLL is present;
		// machines without the shim (or without a Vulkan device, where ggml
		// falls back to its CPU backend) still get the Odin kernels.
		if transcribe_available() { return .Shim }
		return .Phonon
	}
	return .Shim
}

// asr_backend_parse maps a setting to an engine; "auto" and unknown values
// fall through to the family rule.
asr_backend_parse :: proc(s: string) -> (Asr_Backend, bool) {
	switch s {
	case "shim":
		return .Shim, true
	case "phonon", "odin":
		return .Phonon, true
	}
	return .Shim, false
}

// asr_engine_wanted reports whether the setting pins an engine, for `hex engine`.
asr_engine_setting :: proc() -> string {
	cfg := config_load()
	if len(cfg.engine) == 0 { return "auto" }
	return cfg.engine
}

engine_name :: proc(b: Asr_Backend) -> string {
	return b == .Phonon ? "phonon (Odin kernels)" : "shim (transcribe.cpp)"
}

// One live session at a time: the model the config points at. The engines are
// never both open, because each one holds the whole 700 MB mapping.
ph: phonon.Engine
ph_live: bool

phonon_open :: proc(path: string, threads: int) -> (ok: bool, err: string) {
	if ph_live { return true, "" }
	e, open_err := phonon.engine_open(path, threads)
	if open_err != "" { return false, fmt.aprintf("phonon engine: %s", open_err) }
	ph = e
	if gpu_ok, gpu_err := phonon.engine_gpu_status(&ph); gpu_err != "" {
		log_write("[hex] phonon: GPU encoder disabled: %s", gpu_err)
	} else if gpu_ok {
		log_write("[hex] phonon: GPU encoder enabled (HEX_GPU)")
	}
	ph_live = true
	return true, ""
}

phonon_close :: proc() {
	if !ph_live { return }
	phonon.engine_close(&ph)
	ph_live = false
}

// phonon_transcribe returns the decoded text, owned by the caller, for mono
// 16 kHz samples. Parakeet is English-only, so the language argument the shim
// takes has no equivalent here.
phonon_transcribe :: proc(samples: []f32) -> (text: string, err: string) {
	res := phonon.transcribe(&ph, samples)
	if res.text != "" {
		log_write("[hex] phonon: mel %.1f ms, encoder %.1f ms, decoder %.1f ms, %d tokens",
			res.mel_ms, res.enc_ms, res.dec_ms, len(res.tokens))
	}
	if res.tokens != nil { delete(res.tokens) }
	if res.text == "" { return "", strings.clone("phonon engine returned no text") }
	return res.text, ""
}
