package main

import "base:runtime"
import "core:fmt"
import "core:os"
import "core:strings"
import "core:sync"
import "core:thread"
import "core:time"

import "core:sys/windows"

// Dot scope: the HUD from Hex's src/rune_hud.wgsl, rasterized on the CPU.
// Geometry, palette, and compositing follow the shader: a 112x26 pt plate with
// a 43x9 dot field at 2.6 pt pitch, a soft shadow 3 pt below, a 1 px rim and a
// centre glow. Colours are sRGB, premultiplied, composited source-over.

MOD_ALT :: 0x0001
MOD_CONTROL :: 0x0002

WINDOW_W :: 304 // pt, matches Hex's non-macOS HUD window
WINDOW_H :: 80
PITCH :: 2.6

BLENDFUNCTION :: struct {
	BlendOp:             u8,
	BlendFlags:          u8,
	SourceConstantAlpha: u8,
	AlphaFormat:         u8,
}
ULW_ALPHA :: 0x02
ERROR_ALREADY_EXISTS :: 183
AC_SRC_ALPHA :: 0x01

// Id of the RT_GROUP_ICON in hex.res (see hex.rc / scripts/make-icon.py).
RES_ICON :: 1

// Tray callback message: WM_APP + 1, delivered to the HUD window when the
// notification-area icon is clicked.
WM_TRAY :: 0x8000 + 1
// A transcription worker asks the GUI thread to drop Escape's reservation.
// UnregisterHotKey only works on the thread that registered the hotkey, and
// the worker is not it (a worker-thread call fails with
// ERROR_HOTKEY_NOT_REGISTERED and leaves Escape captured for every other app).
WM_ESC_RELEASE :: 0x8000 + 2
// NOTIFYICON_VERSION_4 from shellapi.h: wparam low word = notification, lparam
// low word = the mouse message that produced it.
NOTIFYICON_VERSION_4 :: 4
NIN_SELECT :: 0x0400

foreign import user32 "system:user32.lib"
@(default_calling_convention = "system")
foreign user32 {
	UpdateLayeredWindow :: proc(hwnd: windows.HWND, hdcDst: windows.HDC, pptDst: ^windows.POINT, psize: ^windows.SIZE, hdcSrc: windows.HDC, pptSrc: ^windows.POINT, crKey: windows.COLORREF, pblend: ^BLENDFUNCTION, dwFlags: windows.DWORD) -> windows.BOOL ---
	GetDpiForWindow   :: proc(hwnd: windows.HWND) -> windows.UINT ---
}

foreign import kernel32 "system:kernel32.lib"
@(default_calling_convention = "system")
foreign kernel32 {
	CreateMutexW     :: proc(attrs: rawptr, initial_owner: windows.BOOL, name: [^]u16) -> windows.HANDLE ---
	GetConsoleWindow :: proc() -> windows.HWND ---
}

Hud :: struct {
	hwnd:      windows.HWND,
	mem_dc:    windows.HDC,
	bitmap:    windows.HBITMAP,
	pixels:    []u32, // DIB, the size Windows composites
	w:         int,
	h:         int,
	scale:     f32,
	recording:  bool,
	capture_up: bool,
	busy:       bool,
	cancel_gen: int,
	state_lock:  sync.Mutex,
	buffer_lock: sync.Mutex,

	timer_on:  bool,
	visible:   bool,
	reveal:    f32, // 0 hidden .. 1 shown
	buffer:    ^[dynamic]f32,
	last:      time.Time,
	ready:     time.Time,
	scope_ind:  ^Indicator,
	scope_origin: time.Tick,
	scope_last: f64,
	was_busy:   bool,
	cap_level:  Capture_Level,
	level_debug_last: f64,
}

hud: Hud

app_run :: proc(cfg: ^Config) {
	_ = cfg
	// One HUD per desktop: a second launch (double-click again) exits instead of
	// stacking hotkeys and duplicate paste workers.
	mutex_name := windows.utf8_to_utf16("Local/hex-odin-single-instance", context.temp_allocator)
	if CreateMutexW(nil, windows.BOOL(true), raw_data(mutex_name)) == nil ||
	   windows.GetLastError() == ERROR_ALREADY_EXISTS {
		fmt.println("hex is already running")
		os.exit(0)
	}
	windows.SetProcessDpiAwarenessContext(windows.DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)
	instance := windows.GetModuleHandleW(nil)

	wc: windows.WNDCLASSEXW
	wc.cbSize = size_of(windows.WNDCLASSEXW)
	wc.style = windows.CS_HREDRAW | windows.CS_VREDRAW
	wc.lpfnWndProc = wnd_proc
	wc.hInstance = windows.HANDLE(instance)
	// Icon id 1 comes from hex.res (scripts/make-icon.py + rc.exe), which
	// build.ps1 links with -resource. A build without the resource loads nil,
	// which just leaves the default window icon. The class icon is what the
	// taskbar button and Alt-Tab draw.
	icon_big := windows.HICON(windows.LoadIconW(windows.HINSTANCE(instance), cast(cstring16)windows.MAKEINTRESOURCEW(RES_ICON)))
	icon_small := windows.HICON(windows.LoadImageW(
		windows.HINSTANCE(instance),
		cast(cstring16)windows.MAKEINTRESOURCEW(RES_ICON),
		windows.IMAGE_ICON,
		windows.GetSystemMetrics(windows.SM_CXSMICON),
		windows.GetSystemMetrics(windows.SM_CYSMICON),
		windows.LR_DEFAULTCOLOR,
	))
	if icon_big != nil { wc.hIcon = icon_big }
	if icon_small != nil { wc.hIconSm = icon_small }
	cls := windows.utf8_to_utf16("HexOdinHud", context.temp_allocator)
	wc.lpszClassName = cast(cstring16)raw_data(cls)
	if windows.RegisterClassExW(&wc) == 0 {
		fmt.eprintln("RegisterClassEx failed:", windows.GetLastError())
		os.exit(1)
	}

	screen_w := windows.GetSystemMetrics(windows.SM_CXSCREEN)
	x := (screen_w - WINDOW_W) / 2
	y := 12 // TOP_OFFSET in Hex: the HUD hangs below the top edge

	style := windows.WS_POPUP | windows.WS_VISIBLE
	// WS_EX_TOOLWINDOW keeps the overlay out of the taskbar and Alt-Tab. hex
	// advertises itself through the notification-area icon instead, so the button
	// is not duplicated in the taskbar.
	ex_style := windows.WS_EX_LAYERED | windows.WS_EX_TOPMOST | windows.WS_EX_TOOLWINDOW | windows.WS_EX_NOACTIVATE | windows.WS_EX_TRANSPARENT
	h := windows.CreateWindowExW(ex_style, wc.lpszClassName, cast(cstring16)raw_data(windows.utf8_to_utf16("Hex", context.temp_allocator)), style, i32(x), i32(y), WINDOW_W, WINDOW_H, nil, nil, windows.HANDLE(instance), nil)
	if h == nil {
		fmt.eprintln("CreateWindowEx failed:", windows.GetLastError())
		os.exit(1)
	}
	// Belt and braces: set both icons on the window as well, so the button keeps
	// the mark even if the class was registered by an earlier code path.
	ICON_SMALL :: 0
	ICON_BIG :: 1
	if icon_big != nil { windows.SendMessageW(h, windows.WM_SETICON, ICON_BIG, windows.LPARAM(uintptr(icon_big))) }
	if icon_small != nil { windows.SendMessageW(h, windows.WM_SETICON, ICON_SMALL, windows.LPARAM(uintptr(icon_small))) }
	hud.hwnd = h
	hud.ready = time.now()

	// Hex keeps the capture stream open for the life of the app, so pressing the
	// hotkey starts recording instantly instead of waiting on device init.
	if vad_load() {
		log_write("[hex] silero vad loaded")
	} else {
		log_write("[hex] vad unavailable, transcribing without silence gate")
	}
	// The mic is opened per dictation and released when it ends, so Windows
	// does not show the microphone in use while hex is just sitting there.
	hud.capture_up = false
	idle_thread = thread.create(model_idle_worker)
	if idle_thread != nil {
		idle_thread.flags += {.Self_Cleanup}
		thread.start(idle_thread)
	} else {
		log_write("[hex] could not start model idle worker")
	}
	hud.scale = 1.0
	if dpi := f32(GetDpiForWindow(h)); dpi >= 96 { hud.scale = dpi / 96.0 }
	hud_setup_dib()
	hud.scope_ind = indicator_new(tuning_default())
	hud.scope_origin = time.tick_now()
	hud.scope_last = 0
	hud.reveal = 0
	hud.last = time.now()

	if !windows.RegisterHotKey(h, 1, MOD_CONTROL, u32(windows.VK_SPACE)) {
		fmt.eprintln("Ctrl+Space is taken by another app; rebind it there first")
	}
	windows.RegisterHotKey(h, 2, MOD_CONTROL | MOD_ALT, u32('Q'))
	tray_add(h)
	hud_timer(true)

	fmt.println("HUD up — Ctrl+Space dictate, Ctrl+Alt+Q quit")
	msg: windows.MSG
	for windows.GetMessageW(&msg, nil, 0, 0) > 0 {
		windows.TranslateMessage(&msg)
		windows.DispatchMessageW(&msg)
	}
}

// Notification-area icon: the only always-visible affordance while hex is
// resident, since the HUD window is a tool window that hides itself when idle.
tray_icon: windows.HICON
tray_installed: bool

tray_add :: proc(hwnd: windows.HWND) {
	instance := windows.GetModuleHandleW(nil)
	ic := windows.LoadImageW(
		windows.HINSTANCE(instance),
		cast(cstring16)windows.MAKEINTRESOURCEW(RES_ICON),
		windows.IMAGE_ICON,
		windows.GetSystemMetrics(windows.SM_CXSMICON),
		windows.GetSystemMetrics(windows.SM_CYSMICON),
		windows.LR_DEFAULTCOLOR,
	)
	if ic == nil {
		log_write("[hex] tray icon missing from hex.res")
		return
	}
	tray_icon = windows.HICON(ic)
	nid: windows.NOTIFYICONDATAW
	nid.cbSize = u32(size_of(windows.NOTIFYICONDATAW))
	nid.hWnd = hwnd
	nid.uID = 1
	nid.uFlags = windows.NIF_MESSAGE | windows.NIF_ICON | windows.NIF_TIP | windows.NIF_SHOWTIP
	nid.uCallbackMessage = WM_TRAY
	nid.hIcon = tray_icon
	tip := windows.utf8_to_utf16("hex — Ctrl+Space to dictate", context.temp_allocator)
	for ch, i in tip {
		if i >= len(nid.szTip) - 1 { break }
		nid.szTip[i] = ch
	}
	if !windows.Shell_NotifyIconW(windows.NIM_ADD, &nid) {
		log_write("[hex] could not add the notification-area icon")
		return
	}
	nid.uVersion = NOTIFYICON_VERSION_4
	windows.Shell_NotifyIconW(windows.NIM_SETVERSION, &nid)
	tray_installed = true
}

tray_remove :: proc() {
	if !tray_installed || hud.hwnd == nil { return }
	nid: windows.NOTIFYICONDATAW
	nid.cbSize = u32(size_of(windows.NOTIFYICONDATAW))
	nid.hWnd = hud.hwnd
	nid.uID = 1
	windows.Shell_NotifyIconW(windows.NIM_DELETE, &nid)
	tray_installed = false
}

tray_menu :: proc(hwnd: windows.HWND) {
	menu := windows.CreatePopupMenu()
	if menu == nil { return }
	quit := windows.utf8_to_utf16("Quit hex", context.temp_allocator)
	windows.AppendMenuW(menu, windows.MF_STRING, 1, cast(cstring16)raw_data(quit))
	pt: windows.POINT
	windows.GetCursorPos(&pt)
	// The owning window must be foreground or the menu will not close when the
	// click lands elsewhere.
	windows.SetForegroundWindow(hwnd)
	cmd := windows.TrackPopupMenu(menu, windows.TPM_RIGHTBUTTON | windows.TPM_RETURNCMD, pt.x, pt.y, 0, hwnd, nil)
	windows.DestroyMenu(menu)
	windows.PostMessageW(hwnd, 0 /* WM_NULL */, 0, 0)
	if cmd == 1 { windows.PostQuitMessage(0) }
}

wnd_proc :: proc "system" (hwnd: windows.HWND, msg: windows.UINT, wparam: windows.WPARAM, lparam: windows.LPARAM) -> windows.LPARAM {
	context = runtime.default_context()
	switch msg {
	case windows.WM_TIMER:
		hud_tick()
		return 0
	case WM_TRAY:
		// Version-4 layout: wparam low word is the notification, lparam low word
		// is the mouse message. Right click or the keyboard menu key opens the
		// context menu; a left click does nothing (the HUD is the surface).
		note := u16(u64(wparam) & 0xffff)
		mouse := u16(u64(lparam) & 0xffff)
		if note == windows.WM_CONTEXTMENU || mouse == windows.WM_RBUTTONUP || note == 0x0004 {
			tray_menu(hwnd)
		}
		return 0
	case windows.WM_HOTKEY:
		switch wparam {
		case 1: hud_toggle()
		case 2: windows.PostQuitMessage(0)
		case 3: hud_cancel()
		}
		return 0
	case WM_ESC_RELEASE:
		// A dictation that began after the worker posted this owns Escape; its
		// own end releases it.
		if !hud.recording && !hud_is_busy() {
			hud_escape(false)
		}
		return 0
	case windows.WM_DESTROY:
		// The stream outlived every recording; close it with the app.
		sync.atomic_store(&pump_running, i32(0))
		if hud.capture_up { capture_stop(nil) }
		tray_remove()
		windows.PostQuitMessage(0)
		return 0
	}
	return windows.DefWindowProcW(hwnd, msg, wparam, lparam)
}

hud_setup_dib :: proc() {
	hud.w = int(f32(WINDOW_W) * hud.scale)
	hud.h = int(f32(WINDOW_H) * hud.scale)

	bi: windows.BITMAPINFO
	bi.bmiHeader.biSize = size_of(windows.BITMAPINFOHEADER)
	bi.bmiHeader.biWidth = i32(hud.w)
	bi.bmiHeader.biHeight = -i32(hud.h)
	bi.bmiHeader.biPlanes = 1
	bi.bmiHeader.biBitCount = 32
	bi.bmiHeader.biCompression = windows.BI_RGB

	screen_dc := windows.GetDC(nil)
	defer windows.ReleaseDC(nil, screen_dc)
	hud.mem_dc = windows.CreateCompatibleDC(screen_dc)
	bits: rawptr
	hud.bitmap = windows.CreateDIBSection(hud.mem_dc, &bi, windows.DIB_RGB_COLORS, &bits, nil, 0)
	if hud.bitmap == nil || bits == nil {
		fmt.eprintln("CreateDIBSection failed:", windows.GetLastError())
		os.exit(1)
	}
	windows.SelectObject(hud.mem_dc, windows.HGDIOBJ(hud.bitmap))
	p32 := cast([^]u32)bits
	hud.pixels = p32[0:hud.w * hud.h]
}

HUD_TIMER_MS :: 42 // 24 fps; the braid does not need more and this is the
                  // difference between 12% and 5% of a core while recording.

hud_timer :: proc(on: bool) {
	if hud.hwnd == nil { return }
	if on {
		if !hud.timer_on {
			windows.SetTimer(hud.hwnd, 1, HUD_TIMER_MS, nil)
			hud.timer_on = true
		}
	} else if hud.timer_on {
		windows.KillTimer(hud.hwnd, 1)
		hud.timer_on = false
	}
}

hud_toggle :: proc() {
	if !hud.recording {
		if hud_is_busy() {
			log_write("[hex] busy; ignoring hotkey")
			return
		}
		if !hud.capture_up {
			if _, err := capture_start(); err != .None {
				log_write("[hex] mic open failed: %v", err)
				fmt.eprintln("[hex] no mic:", err)
				return
			}
			hud.capture_up = true
			log_write("[hex] mic opened")
		}
		sync.lock(&hud.buffer_lock)
		if hud.buffer == nil { hud.buffer = new([dynamic]f32) }
		clear(&hud.buffer^)
		hud.cap_level = {}
		sync.unlock(&hud.buffer_lock)
		capture_discard()
		hud_set_pump(true)
		hud.recording = true
		hud.reveal = 1
		if hud.scope_ind != nil {
			now := time.duration_seconds(time.tick_since(hud.scope_origin))
			indicator_on_started(hud.scope_ind, now)
		}
		hud_timer(true)
		power_eco(false)
		hud_escape(true)
		log_write("[hex] listening")
		// 890 ms of model init and 7.5 ms of VAD init overlap the recording
		// instead of landing between the key release and the paste.
		preload_model_async()
		if segmental_enabled() {
			seg_cursor = 0
			seg_model = config_load().model
			clear(&seg_texts)
			sync.atomic_store(&seg_stop, i32(0))
			seg_thread = thread.create(seg_worker)
			if seg_thread != nil {
				thread.start(seg_thread)
			}
		}
	} else {
		hud.recording = false
		hud_set_pump(false)
		sync.lock(&hud.buffer_lock)
		capture_pump(nil, hud.buffer, &hud.cap_level) // sweep the final period
		sync.unlock(&hud.buffer_lock)
		capture_stop(nil)
		hud.capture_up = false
		log_write("[hex] mic released, transcribing")
		if hud.scope_ind != nil {
			now := time.duration_seconds(time.tick_since(hud.scope_origin))
			indicator_on_submitted(hud.scope_ind, now)
		}
		hud_transcribe_async()
	}
}

pump_thread: ^thread.Thread
pump_running: i32
idle_thread: ^thread.Thread

// The callback keeps the ring filled for the life of the app; only a recording
// drains it. Idle wakeups are the expensive part on a laptop, not the copy.
hud_set_pump :: proc(on: bool) {
	if on {
		if pump_thread != nil { return }
		sync.atomic_store(&pump_running, i32(1))
		pump_thread = thread.create(proc(_: ^thread.Thread) {
			for sync.atomic_load(&pump_running) != 0 {
				sync.lock(&hud.buffer_lock)
				if hud.buffer != nil { capture_pump(nil, hud.buffer, &hud.cap_level) }
				sync.unlock(&hud.buffer_lock)
				time.sleep(10 * time.Millisecond)
			}
		})
		thread.start(pump_thread)
	} else {
		if pump_thread == nil { return }
		sync.atomic_store(&pump_running, i32(0))
		thread.join(pump_thread)
		thread.destroy(pump_thread)
		pump_thread = nil
	}
}

// Escape is only reserved while a dictation can still be cancelled, so it stays
// available to every other app the rest of the time.
hud_escape :: proc(on: bool) {
	if hud.hwnd == nil { return }
	if on {
		windows.RegisterHotKey(hud.hwnd, 3, 0, 0x1B)
	} else {
		windows.UnregisterHotKey(hud.hwnd, 3)
	}
}

hud_cancel :: proc() {
	if hud.recording {
		hud.recording = false
		hud_set_pump(false)
		capture_stop(nil)
		hud.capture_up = false
		sync.lock(&hud.buffer_lock)
		if hud.buffer != nil { clear(&hud.buffer^) }
		sync.unlock(&hud.buffer_lock)
		sync.lock(&hud.state_lock)
		hud.cancel_gen += 1
		sync.unlock(&hud.state_lock)
		hud_escape(false)
		if hud.scope_ind != nil {
			hud.scope_ind.capturing = false
			hud.scope_ind.pending_jobs = 0
			hud.scope_ind.completion_pending = false
			hud.scope_ind.phase = .cancelled
			hud.scope_ind.phase_started = time.duration_seconds(time.tick_since(hud.scope_origin))
		}
		log_write("[hex] cancelled recording")
	} else if hud_is_busy() {
		sync.lock(&hud.state_lock)
		hud.cancel_gen += 1
		sync.unlock(&hud.state_lock)
		if hud.scope_ind != nil {
			hud.scope_ind.phase = .cancelled
			hud.scope_ind.pending_jobs = 0
			hud.scope_ind.completion_pending = false
		}
		log_write("[hex] cancelling transcription")
	}
}

hud_is_busy :: proc() -> bool {
	sync.lock(&hud.state_lock)
	defer sync.unlock(&hud.state_lock)
	return hud.busy
}

hud_cancel_generation :: proc() -> int {
	sync.lock(&hud.state_lock)
	defer sync.unlock(&hud.state_lock)
	return hud.cancel_gen
}

// --- Segmental dictation ------------------------------------------------
// While the user is still talking, transcribe every speech segment the VAD
// has already closed, so key-up only has to encode the tail. The encoder is
// ~O(seconds), so moving finished audio off the critical path is the only
// change here that shortens a long dictation rather than a short one.
//
// On by default; HEX_SEGMENTAL=0 disables it. Splice points come from the VAD,
// so they cannot cut a word in half, and each slice carries a little left
// context anyway.
SEG_MIN_SAMPLES :: 16000 // 1 s of new speech before a slice is worth encoding
SEG_CTX_SAMPLES :: 3200  // 200 ms of left context per slice
SEG_POLL_MS :: 500    // poll the VAD every 500 ms
SEG_SILENCE_MS :: 500    // silence that closes a segment

seg_thread: ^thread.Thread
seg_stop: i32
seg_cursor: int
seg_model: string
seg_texts: [dynamic]string

segmental_enabled :: proc() -> bool {
	if shim.vad_speech_end == nil { return false }
	// On by default; HEX_SEGMENTAL=0 restores the encode-everything-at-key-up
	// path, and a shim without hex_vad_speech_end disables it automatically.
	return os.get_env("HEX_SEGMENTAL", context.temp_allocator) != "0"
}

seg_worker :: proc(_: ^thread.Thread) {
	for sync.atomic_load(&seg_stop) == 0 {
		time.sleep(SEG_POLL_MS * time.Millisecond)
		if sync.atomic_load(&seg_stop) != 0 { break }

		end := 0
		slice: []f32
		sync.lock(&hud.buffer_lock)
		if hud.buffer != nil {
			buf := hud.buffer^
			if len(buf) > SEG_CTX_SAMPLES {
				end = int(shim.vad_speech_end(raw_data(buf), i32(len(buf)), SEG_SILENCE_MS))
			}
			if end > seg_cursor+SEG_MIN_SAMPLES && end <= len(buf) {
				lo := max(0, seg_cursor-SEG_CTX_SAMPLES)
				slice = make([]f32, end-lo)
				copy(slice, buf[lo:end])
			}
		}
		sync.unlock(&hud.buffer_lock)
		if slice == nil { continue }

		text, err := transcribe_run(seg_model, slice)
		delete(slice)
		if err != "" {
			delete(err)
		} else if len(text) > 0 {
			append(&seg_texts, text)
		}
		seg_cursor = end
	}
}

TranscribeJob :: struct {
	audio: []f32,
	gen:   int,
}

hud_transcribe_async :: proc() {
	sync.lock(&hud.buffer_lock)
	if hud.buffer == nil || len(hud.buffer^) < 1600 { // < 100 ms
		sync.unlock(&hud.buffer_lock)
		hud_escape(false)
		log_write("[hex] too short")
		return
	}
	// The recording buffer is reused by the next recording; the transcription
	// gets its own copy so a new one cannot move memory under it. When the
	// background segmenter is running the worker stops it first and copies
	// only the tail, so the copy is deferred to it.
	audio: []f32
	if seg_thread == nil {
		audio = make([]f32, len(hud.buffer^))
		copy(audio, hud.buffer^[:])
	}
	sync.unlock(&hud.buffer_lock)

	job := new(TranscribeJob)
	if job == nil {
		delete(audio)
		hud_escape(false)
		log_write("[hex] out of memory allocating transcription job")
		return
	}
	job.audio = audio

	sync.lock(&hud.state_lock)
	if hud.busy {
		sync.unlock(&hud.state_lock)
		delete(audio)
		free(job)
		log_write("[hex] busy; ignoring hotkey")
		return
	}
	job.gen = hud.cancel_gen
	hud.busy = true
	sync.unlock(&hud.state_lock)
	hud_escape(true)

	t := thread.create(transcribe_worker)
	if t == nil {
		hud_escape(false)
		sync.lock(&hud.state_lock)
		hud.busy = false
		sync.unlock(&hud.state_lock)
		delete(audio)
		free(job)
		log_write("[hex] could not create transcription thread")
		return
	}
	t.data = job
	t.flags += {.Self_Cleanup}
	thread.start(t)
}

transcribe_worker :: proc(t: ^thread.Thread) {
	job := cast(^TranscribeJob)t.data
	gen := job.gen
	cfg := config_load()
	audio := job.audio
	// Segmental: stop the background segmenter, then encode only the tail.
	// The join waits at most one in-flight segment, which is the whole point:
	// that work already happened while the user was still talking.
	prefix: []string
	if seg_thread != nil {
		sync.atomic_store(&seg_stop, i32(1))
		thread.join(seg_thread)
		thread.destroy(seg_thread)
		seg_thread = nil
	}
	prefix = seg_texts[:]
	if audio == nil {
		sync.lock(&hud.buffer_lock)
		if hud.buffer != nil && seg_cursor < len(hud.buffer^) {
			audio = make([]f32, len(hud.buffer^)-seg_cursor)
			copy(audio, hud.buffer^[seg_cursor:])
		}
		sync.unlock(&hud.buffer_lock)
		job.audio = audio
	}
	defer {
		// The dictation's lines (paste text, errors) go out before the
		// worker disappears, so a diagnostic read afterwards is complete.
		log_flush()
		// The GUI thread owns the hotkey, so the worker asks it to release.
		windows.PostMessageW(hud.hwnd, WM_ESC_RELEASE, 0, 0)
		sync.lock(&hud.state_lock)
		hud.busy = false
		sync.unlock(&hud.state_lock)
		delete(job.audio)
		free(job)
	}

	start := time.now()
	if len(prefix) == 0 && os.get_env("HEX_SKIP_VAD", context.temp_allocator) == "" && !vad_has_speech(audio) {
		log_write("[hex] no speech")
		return
	}
	if audio == nil || len(audio) == 0 {
		// Every second already went out as a closed segment; nothing to add.
		if len(prefix) > 0 && hud_cancel_generation() == gen {
			full := strings.join(prefix, " ")
			defer delete(full)
			if paste_text(full) {
				log_write("[hex] pasted segmental %.2fs: %s", time.duration_seconds(time.since(start)), full)
			}
		}
		return
	}
	text, err := transcribe_run(cfg.model, audio)
	elapsed := time.duration_seconds(time.since(start))
	if err != "" {
		log_write("[hex] error: %s", err)
		delete(err)
		return
	}
	defer delete(text)
	if hud_cancel_generation() != gen {
		log_write("[hex] cancelled before paste")
		return
	}
	// Closed segments first, then the tail, as one paste in spoken order.
	full := text
	if len(prefix) > 0 {
		parts := make([dynamic]string, 0, len(prefix)+1, context.temp_allocator)
		for p in prefix { append(&parts, p) }
		if len(text) > 0 { append(&parts, text) }
		full = strings.join(parts[:], " ")
	}
	// `defer` is scope-scoped in Odin: putting this inside the if-block above
	// freed `full` before the paste and the log line ever read it, so the
	// clipboard received a recycled heap block (first word replaced by the
	// allocator's free-list pointer). Delete it once both readers are done.
	if paste_text(full) {
		log_write("[hex] pasted %.2fs: %s", elapsed, full)
	} else {
		log_write("[hex] paste failed")
	}
	if len(prefix) > 0 { delete(full) }
}

hud_tick :: proc() {
	now := time.now()
	dt := f32(time.duration_seconds(time.since(hud.last)))
	if dt > 0.1 { dt = 0.1 }
	hud.last = now
	scope_now := time.duration_seconds(time.tick_since(hud.scope_origin))
	scope_dt := f32(scope_now - hud.scope_last)
	hud.scope_last = scope_now

	// Level: the raw period RMS the capture pump already measured. No second
	// derivation from the buffer, and no idle floor overriding quiet speech.
	raw_rms := f32(0)
	raw_peak := f32(0)
	if hud.recording {
		sync.lock(&hud.buffer_lock)
		raw_rms = hud.cap_level.rms
		raw_peak = hud.cap_level.peak
		sync.unlock(&hud.buffer_lock)
	}
	busy := hud_is_busy()

	raw_db := loudness_dbfs(raw_rms)
	if hud.scope_ind != nil {
		// The indicator contract is dBFS: src/dictation/indicator.rs sends
		// 20*log10(rms), and AutoLevel steps in dB.
		hud.scope_ind.loudness = raw_db
		if hud.was_busy && !busy && !hud.recording {
			indicator_on_completed(hud.scope_ind, scope_now)
		}
		if hud.recording && os.get_env("HEX_LEVEL_DEBUG", context.temp_allocator) != "" &&
		   scope_now - hud.level_debug_last >= 0.1 {
			hud.level_debug_last = scope_now
			auto := &hud.scope_ind.scope.auto
			log_write("[hex] level rms=%.6f peak=%.6f dbfs=%.2f scope=%.2f auto_peak=%.2f auto_room=%.2f",
				raw_rms, raw_peak, raw_db, hud.scope_ind.loudness,
				auto.peak, auto.room)
		}
	}
	hud.was_busy = busy

	// The HUD only exists while dictating: fade in on press, out after paste.
	want_reveal: f32 = (hud.recording || busy || time.since(hud.ready) < 1200 * time.Millisecond) ? 1 : 0
	hud.reveal += (want_reveal - hud.reveal) * min(1, dt * (want_reveal > hud.reveal ? 14 : 7))
	if hud.reveal < 0.01 && want_reveal == 0 {
		if hud.visible {
			windows.ShowWindow(hud.hwnd, windows.SW_HIDE)
			hud.visible = false
			power_eco(true)
		}
		hud_timer(false)
		return
	}
	if !hud.visible {
		windows.ShowWindow(hud.hwnd, windows.SW_SHOWNOACTIVATE)
		hud.visible = true
	}

	if hud.scope_ind != nil {
		frame := indicator_step(hud.scope_ind, scope_now, scope_dt)
		hud_render_scope(frame)
	}
	hud_blit()
}

hud_blit :: proc() {
	p: windows.POINT
	p.x = i32((int(windows.GetSystemMetrics(windows.SM_CXSCREEN)) - int(f32(WINDOW_W) * hud.scale)) / 2)
	p.y = 12
	size: windows.SIZE
	size.cx = i32(hud.w)
	size.cy = i32(hud.h)
	src: windows.POINT
	blend: BLENDFUNCTION
	blend.BlendOp = 0
	blend.SourceConstantAlpha = 255
	blend.AlphaFormat = AC_SRC_ALPHA
	UpdateLayeredWindow(hud.hwnd, nil, &p, &size, hud.mem_dc, &src, 0, &blend, ULW_ALPHA)
}

clamp :: proc(v, lo, hi: f32) -> f32 {
	if v < lo { return lo }
	if v > hi { return hi }
	return v
}

