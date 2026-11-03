package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:sync"
import "core:time"

// Same data root as the Rust build (%APPDATA%\hex) so models, history and
// settings are shared while both implementations coexist.
Config :: struct {
	data_dir:   string,
	models_dir: string,
	logs_dir:   string,
	model:      string,
	language:   string,
	shortcut:   string,
	paste_keys: string,
	threads:    int,
	engine:     string,
	path:       string,
}

DEFAULTS :: Config{
	model = "phonon25_en",
	language = "en",
	shortcut = "Ctrl+Space",
	paste_keys = "ctrl_v",
	threads = 0, // 0 = let ggml use every usable core
	engine = "auto", // auto = built-in engine for phonon models, shim elsewhere
}

// config_load is called by log_write on every log line, so it must not allocate
// per call. The parsed config is cached and only re-read when the file's mtime
// or presence changes; config_save refreshes the cache after writing.
//
// The mtime probe is itself a syscall (~55 us here), so it is rate limited to
// one check per CONFIG_RECHECK. The app's own writes go through config_save and
// are visible immediately; the window only bounds how late an edit made by
// another process is noticed, which no hot path depends on.
CONFIG_RECHECK :: 250 * time.Millisecond

config_lock: sync.Mutex
cached_config: Config
config_cached: bool
config_path: string
config_file_present: bool
config_file_time: time.Time
config_checked: time.Tick

config_file_stamp :: proc(path: string) -> (present: bool, mtime: time.Time) {
	t, err := os.modification_time_by_path(path)
	return err == nil, t
}

config_file_changed :: proc() -> bool {
	if len(config_path) == 0 { return true }
	if time.tick_since(config_checked) < CONFIG_RECHECK { return false }
	present, mtime := config_file_stamp(config_path)
	config_checked = time.tick_now()
	if present != config_file_present { return true }
	return present && mtime != config_file_time
}

config_load :: proc() -> Config {
	sync.lock(&config_lock)
	defer sync.unlock(&config_lock)
	if config_cached && !config_file_changed() {
		return cached_config
	}

	cfg := DEFAULTS
	appdata := os.get_env("APPDATA", context.temp_allocator)
	if len(appdata) == 0 { appdata = "." }
	cfg.data_dir = strings.concatenate({appdata, "\\hex"})
	cfg.models_dir = strings.concatenate({cfg.data_dir, "\\models"})
	cfg.logs_dir = strings.concatenate({cfg.data_dir, "\\logs"})
	cfg.path = strings.concatenate({cfg.data_dir, "\\odin-config"})

	data, err := os.read_entire_file(cfg.path, context.temp_allocator)
	if err == nil {
		for line in strings.split_lines(string(data)) {
			kv := strings.split_n(line, "=", 2)
			if len(kv) != 2 { continue }
			key := strings.trim_space(kv[0])
			val := strings.trim_space(kv[1])
			switch key {
			case "model":    cfg.model = strings.clone(val)
			case "language": cfg.language = strings.clone(val)
			case "shortcut": cfg.shortcut = strings.clone(val)
			case "paste_keys": cfg.paste_keys = strings.clone(val)
			case "threads":
				if n, ok := strconv.parse_int(val); ok { cfg.threads = n }
			case "engine":
				switch val {
				case "auto", "shim", "phonon", "odin": cfg.engine = strings.clone(val)
				}
			}
		}
	}
	// A stale config from an older registry should not brick startup.
	if model_by_id(cfg.model) == nil { cfg.model = DEFAULTS.model }
	config_path = cfg.path
	config_file_present, config_file_time = config_file_stamp(cfg.path)
	config_checked = time.tick_now()
	cached_config = cfg
	config_cached = true
	return cfg
}

config_save :: proc(cfg: ^Config) -> bool {
	os.make_directory(cfg.data_dir)
	text := fmt.aprintf("model=%s\nlanguage=%s\nshortcut=%s\npaste_keys=%s\nthreads=%d\nengine=%s\n", cfg.model, cfg.language, cfg.shortcut, cfg.paste_keys, cfg.threads, cfg.engine)
	ok := os.write_entire_file(cfg.path, text) == nil
	if ok {
		sync.lock(&config_lock)
		cached_config = cfg^
		config_cached = true
		config_path = cfg.path
		config_file_present, config_file_time = config_file_stamp(cfg.path)
		config_checked = time.tick_now()
		sync.unlock(&config_lock)
	}
	return ok
}
