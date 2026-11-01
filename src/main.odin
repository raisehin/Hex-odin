package main

import "core:fmt"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:sys/windows"
import "core:thread"
import "core:time"

VERSION :: "0.1.0-odin"

main :: proc() {
	// Batched log lines must reach the file even when a command exits early.
	defer log_flush()
	args := os.args
	// Explorer double-click passes no arguments: behave like an app, not a CLI.
	if len(args) <= 1 {
		cmd_desktop()
		return
	}
	cmd := args[1]
	switch cmd {
	case "devices":
		cmd_devices()
	case "record":
		out := len(args) > 2 ? args[2] : "capture.wav"
		cmd_record(out)
	case "transcribe":
		if len(args) < 3 { fmt.eprintln("usage: hex transcribe <file.wav> [model] [language]"); os.exit(2) }
		cmd_transcribe(args[2], len(args) > 3 ? args[3] : "", len(args) > 4 ? args[4] : "")
	case "models":
		cmd_models(args[2:])
	case "app":
		cmd_app()
	case "shortcut":
		cmd_shortcut()
	case "paste":
		cmd_paste(args[2:])
	case "paste-test":
		if len(args) < 3 { fmt.eprintln("usage: hex paste-test <text>"); os.exit(2) }
		if !paste_text(args[2]) { os.exit(1) }
		// Keep the process alive long enough for the clipboard restore thread.
		time.sleep(700 * time.Millisecond)
	case "threads":
		cmd_threads(args[2:])
	case "vad":
		if len(args) < 3 { fmt.eprintln("usage: hex vad <file.wav>"); os.exit(2) }
		cmd_vad(args[2])
	case "status":
		cmd_status()
	case "version", "--version", "-v":
		fmt.println("hex", VERSION)
	case "help", "--help", "-h":
		usage()
	case:
		fmt.eprintln("unknown command:", cmd)
		usage()
		os.exit(2)
	}
}

PASTE_MODES :: [?]string{"ctrl_v", "ctrl_shift_v", "shift_insert", "type"}

cmd_paste :: proc(args: []string) {
	cfg := config_load()
	if len(args) < 1 {
		fmt.println(cfg.paste_keys)
		return
	}
	mode := args[0]
	valid := false
	for m in PASTE_MODES { if m == mode { valid = true } }
	if !valid {
		fmt.eprint("usage: hex paste ctrl_v|ctrl_shift_v|shift_insert|type\n")
		os.exit(2)
	}
	cfg.paste_keys = strings.clone(mode)
	if !config_save(&cfg) { fmt.eprintln("could not save config"); os.exit(1) }
	fmt.println("paste:", mode)
}

cmd_threads :: proc(args: []string) {
	cfg := config_load()
	if len(args) < 1 {
		if cfg.threads == 0 { fmt.println("threads: auto") } else { fmt.println("threads:", cfg.threads) }
		return
	}
	n, ok := strconv.parse_int(args[0])
	if !ok || n < 0 || n > 64 {
		fmt.eprintln("usage: hex threads <0..64>  (0 = all usable cores)")
		os.exit(2)
	}
	cfg.threads = n
	if !config_save(&cfg) { fmt.eprintln("could not save config"); os.exit(1) }
	if n == 0 { fmt.println("threads: auto") } else { fmt.println("threads:", n) }
}

cmd_vad :: proc(path: string) {
	samples, _, ok := wav_read_mono16k(path)
	if !ok { fmt.eprintln("could not read", path); os.exit(1) }
	if !vad_load() { fmt.eprintln("vad model missing (silero-vad-v6.2-F32.gguf)"); os.exit(1) }
	speech := vad_has_speech(samples)
	secs := f64(len(samples)) / 16000.0
	state := speech ? "speech" : "silence"
	fmt.printf("%s: %.2fs -> ", path, secs)
	fmt.println(state)
}

// Re-creates the Desktop / Start Menu shortcuts next to this exe.
cmd_shortcut :: proc() {
	exe := os.args[0]
	sep := strings.last_index_any(exe, "/\\")
	dir := sep >= 0 ? exe[:sep] : "."
	script := fmt.aprintf("%s/scripts/install-shortcut.ps1", dir)
	if !run_ok({"powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script}) {
		fmt.eprintln("shortcut creation failed:", script)
		os.exit(1)
	}
}

// Launched by double-click: no console, no arguments, just the dictation host.
cmd_desktop :: proc() {
	if h := GetConsoleWindow(); h != nil {
		windows.ShowWindow(h, windows.SW_HIDE)
	}
	app_run(nil)
}

usage :: proc() {
	fmt.println(`hex — local dictation (Odin core + Odin UI)

  hex                     (no arguments) start the dictation host, like a double-click
  hex app                 same, explicitly (keeps the console for logs)

While dictating: Ctrl+Space toggles, Esc discards the recording or cancels the
in-flight transcription so nothing is pasted.
  hex shortcut            create Desktop + Start Menu shortcuts
  hex record [file.wav]   capture the default mic until Enter
  hex transcribe <wav>    transcribe a wav (16-bit PCM any rate)
  hex vad <wav>           report silence/speech using the Silero VAD
  hex threads [n]         cap inference CPU workers (0 = all cores)
  hex paste [mode]        ctrl_v | ctrl_shift_v | shift_insert | type
  hex devices             list capture endpoints and compute backends
  hex models [list|install <id>|check <id>]
  hex status              resolved settings and paths
  hex version`)
}

cmd_status :: proc() {
	cfg := config_load()
	fmt.println("hex", VERSION)
	fmt.println("data dir :", cfg.data_dir)
	fmt.println("models   :", cfg.models_dir)
	fmt.println("model    :", cfg.model)
	fmt.println("language :", cfg.language)
	fmt.println("shortcut :", cfg.shortcut)
	fmt.println("paste    :", cfg.paste_keys)
	if cfg.threads == 0 { fmt.println("threads  : auto") } else { fmt.println("threads  :", cfg.threads) }
	if transcribe_available() {
		v := shim.version != nil ? string(shim.version()) : "?"
		fmt.println("shim     : loaded", v)
	} else {
		fmt.println("shim     : not built (scripts/build-native.ps1)")
	}
}

cmd_devices :: proc() {
	caps := capture_devices()
	if len(caps) == 0 { fmt.println("no capture endpoints") }
	for c in caps {
		mark := c.is_default ? " *" : "  "
		fmt.printf("%s%s\n", mark, c.name)
	}
	if transcribe_available() {
		for d in transcribe_devices() { fmt.println("compute:", d) }
	} else {
		fmt.println("compute: shim not built")
	}
}

enter_flag: bool

watch_enter :: proc() {
	b: [1]u8
	for {
		n, err := os.read(os.stdin, b[:])
		if n > 0 { enter_flag = true; return }
		if err != nil { return }
	}
}

cmd_record :: proc(out_path: string) {
	c, err := capture_start()
	if err != .None {
		fmt.eprintln("capture failed:", err)
		os.exit(1)
	}
	defer capture_stop(&c)

	buffer: [dynamic]f32
	level: Capture_Level
	t := thread.create(proc(_: ^thread.Thread) { watch_enter() })
	if t == nil {
		fmt.eprintln("could not start input watcher")
		os.exit(1)
	}
	t.flags += {.Self_Cleanup}
	thread.start(t)

	// Same as the resident app: the model warms up while the user speaks.
	preload_model_async()

	fmt.println("recording — press Enter to stop")
	for !enter_flag {
		capture_pump(&c, &buffer, &level)
		time.sleep(5 * time.Millisecond)
	}

	if !wav_write_f32(out_path, buffer[:], c.rate, 1) {
		fmt.eprintln("could not write", out_path)
		os.exit(1)
	}
	fmt.printf("wrote %s — %d frames @ %d Hz\n", out_path, len(buffer), c.rate)
}

cmd_transcribe :: proc(path: string, model_id := "", language := "") {
	if !transcribe_available() {
		fmt.eprintln("transcribe shim not built — run scripts/build-native.ps1")
		os.exit(1)
	}
	samples, _, ok := wav_read_mono16k(path)
	if !ok {
		fmt.eprintln("could not read", path)
		os.exit(1)
	}
	cfg := config_load()
	use := model_id if len(model_id) > 0 else cfg.model
	if m := model_by_id(use); m == nil {
		fmt.eprintln("unknown model:", use)
		os.exit(2)
	}
	text, err := transcribe_run(use, samples, language)
	if err != "" {
		fmt.eprintln("transcribe:", err)
		os.exit(1)
	}
	fmt.println(text)
}

cmd_models :: proc(args: []string) {
	if len(args) == 0 || args[0] == "list" {
		cfg := config_load()
		for m in MODELS {
			mark := m.id == cfg.model ? "*" : " "
			state := file_exists(model_path(&cfg, m.id)) ? "installed" : "-"
			fmt.printf("%s %-22s %-26s %-4d MB  %-13s %-12s %s\n", mark, m.id, m.name, int(m.bytes / (1024 * 1024)), m.languages, m.realtime, state)
		}
		return
	}
	switch args[0] {
	case "install":
		if len(args) < 2 { fmt.eprintln("usage: hex models install <id>"); os.exit(2) }
		model_install(args[1])
	case "check":
		if len(args) < 2 {
			cfg := config_load()
			for m in MODELS {
				if file_exists(model_path(&cfg, m.id)) { model_check(m.id) }
			}
			return
		}
		model_check(args[1])
	case "use":
		if len(args) < 2 { fmt.eprintln("usage: hex models use <id>"); os.exit(2) }
		if model_by_id(args[1]) == nil { fmt.eprintln("unknown model:", args[1]); os.exit(2) }
		cfg := config_load()
		cfg.model = args[1]
		if config_save(&cfg) { fmt.println("model ->", args[1]) }
	case:
		fmt.eprintln("unknown models subcommand:", args[0])
		os.exit(2)
	}
}

cmd_app :: proc() {
	app_run(nil)
}
