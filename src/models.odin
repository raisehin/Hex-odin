package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:sync"

// GENERATED from cjpais/Handy src-tauri/src/catalog/catalog.json (schema 2).
// Regenerate with scripts/gen_models.py; do not hand-edit the table.
// Adding a model is data: transcribe.cpp reads the architecture out of the GGUF.
Model :: struct {
	id:          string,
	name:        string,
	family:      string,
	description: string,
	repo:        string,
	filename:    string,
	revision:    string,
	bytes:       u64,
	sha256:      string,
	languages:   string,
	realtime:    string,
	streams:     bool,
	curated:     bool,
}

MODELS := []Model{
	{
		id          = "phonon25_en",
		name        = "Phonon-2.5 English Q8_0",
		family      = "parakeet",
		description = "Fermion Phonon-2.5 English, 0.6B TDT. Built locally via scripts/convert-phonon25.py",
		repo        = "",
		filename    = "phonon-2.5-en-Q8_0.gguf",
		revision    = "da200358d638886bc96c758c672ad5568828b119",
		bytes       = 739508608,
		sha256      = "a7c32cae47157a9da95730963f5efa9d6e2f9bd4f7203887fe50ee4ce75aba5f",
		languages   = "en",
		realtime    = "~12x",
		streams     = false,
		curated     = true,
	},
	{
		id          = "canary-180m-flash",
		name        = "Canary 180M Flash",
		family      = "canary",
		description = "Tiny, fast, and runs well on any hardware",
		repo        = "handy-computer/canary-180m-flash-gguf",
		filename    = "canary-180m-flash-Q8_0.gguf",
		revision    = "46b86384d1db59108d6e060a0fc00924cb58499f",
		bytes       = 218447648,
		sha256      = "f942ab746040b531a29ab78ca3afc8dc9564efd8e4c942b5541e53644be9c408",
		languages   = "4 languages",
		realtime    = "~33x",
		streams     = false,
		curated     = true,
	},
}

model_by_id :: proc(id: string) -> ^Model {
	for &m in MODELS { if m.id == id { return &m } }
	return nil
}

// model_path runs once per transcription (transcribe_run) and once per CLI
// install/check. It returns a fresh strings.concatenate that no caller frees,
// which was ~100 B of heap per dictation -- the only unbounded resident-path
// leak left in the audit. The string is now owned here and cached; callers treat
// it as borrowed (process lifetime) and must not delete it.
model_path_lock: sync.Mutex
model_path_cache_dir: string
model_path_cache_id: string
model_path_cache_path: string

model_path :: proc(cfg: ^Config, id: string) -> string {
	m := model_by_id(id)
	if m == nil { return "" }
	sync.lock(&model_path_lock)
	defer sync.unlock(&model_path_lock)
	if len(model_path_cache_path) > 0 &&
	   model_path_cache_dir == cfg.models_dir &&
	   model_path_cache_id == id {
		return model_path_cache_path
	}
	model_path_cache_path = strings.concatenate({cfg.models_dir, "\\", m.filename})
	model_path_cache_dir = cfg.models_dir
	model_path_cache_id = id
	return model_path_cache_path
}

// Downloads through curl.exe (present on Windows 10+) and verifies SHA-256 with
// certutil, so the Odin side needs no HTTP or crypto dependency yet.
model_install :: proc(id: string) {
	m := model_by_id(id)
	if m == nil { fmt.eprintln("unknown model:", id); os.exit(2) }
	cfg := config_load()
	dest := model_path(&cfg, id)
	if file_exists(dest) {
		fmt.println("already installed:", dest)
		return
	}
	if len(m.repo) == 0 {
		// No public GGUF exists in transcribe.cpp's schema for this model.
		fmt.eprintln(m.name, "has no hosted GGUF; it is built from upstream weights:")
		fmt.eprintln("  .\\scripts\\convert-phonon25.py -Install")
		os.exit(1)
	}
	os.make_directory(cfg.models_dir)
	url := fmt.aprintf("https://huggingface.co/%s/resolve/%s/%s?download=true", m.repo, m.revision, m.filename)
	fmt.println("downloading", m.name, "->", dest)
	if !run_ok({"curl.exe", "-fL", "--progress-bar", "-o", dest, url}) {
		fmt.eprintln("download failed")
		os.exit(1)
	}
	if len(m.sha256) == 0 {
		fmt.println("warning: no sha256 in registry for", id)
		return
	}
	actual := sha256_file(dest)
	if actual != m.sha256 {
		fmt.eprintln("sha256 mismatch")
		fmt.eprintln("  expected:", m.sha256)
		fmt.eprintln("  actual  :", actual)
		os.remove(dest)
		os.exit(1)
	}
	fmt.println("verified", dest)
}

model_check :: proc(id: string) {
	m := model_by_id(id)
	if m == nil { fmt.eprintln("unknown model:", id); os.exit(2) }
	cfg := config_load()
	path := model_path(&cfg, id)
	if !file_exists(path) { fmt.println("not installed:", path); return }
	if len(m.sha256) == 0 { fmt.println("installed (no hash to check):", path); return }
	if sha256_file(path) == m.sha256 {
		fmt.println("ok:", path)
	} else {
		fmt.println("CORRUPT:", path)
	}
}

run_ok :: proc(argv: []string) -> bool {
	p, err := os.process_start({command = argv, stdout = os.stdout, stderr = os.stderr})
	if err != nil { return false }
	state, werr := os.process_wait(p)
	if werr != nil { return false }
	return state.exit_code == 0
}

sha256_file :: proc(path: string) -> string {
	out: [dynamic]u8
	if !run_capture({"certutil", "-hashfile", path, "SHA256"}, &out) { return "" }
	text := string(out[:])
	lines := strings.split_lines(text)
	if len(lines) < 2 { return "" }
	line := strings.trim_space(lines[1])
	clean := make([dynamic]u8, context.temp_allocator)
	for ch in line { if ch != ' ' { append(&clean, u8(ch)) } }
	return strings.to_lower(string(clean[:]))
}

run_capture :: proc(argv: []string, out: ^[dynamic]u8) -> bool {
	tmp := "C:/Windows/Temp/hex-capture.txt"
	parts := make([dynamic]string, context.temp_allocator)
	append(&parts, "cmd.exe", "/c")
	joined := strings.join(argv, " ")
	append(&parts, fmt.aprintf("%s > %s", joined, tmp))
	if !run_ok(parts[:]) { return false }
	data, err := os.read_entire_file(tmp, context.temp_allocator)
	os.remove(tmp)
	if err != nil { return false }
	append(out, ..data)
	return true
}

file_exists :: proc(path: string) -> bool {
	_, err := os.stat(path, context.temp_allocator)
	return err == nil
}
