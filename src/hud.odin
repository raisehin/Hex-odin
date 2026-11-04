package main

import "base:runtime"
import "core:fmt"
import "core:math"
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
PLATE_W :: 112.0
PLATE_H :: 26.0
PLATE_RADIUS :: 13.0
COLUMNS :: 43
ROWS :: 9
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

Vec3 :: [3]f32

Hud :: struct {
	hwnd:      windows.HWND,
	mem_dc:    windows.HDC,
	bitmap:    windows.HBITMAP,
	pixels:    []u32, // DIB, the size Windows composites
	hi:        []u32, // supersampled frame
	w:         int,
	h:         int,
	scale:     f32,
	level:     f32, // smoothed rms, 0..1
	cool:      f32, // 0 listening (warm), 1 processing (blue)
	peak:      f32,
	recording:  bool,
	capture_up: bool,
	busy:       bool,
	cancel_gen: int,
	state_lock:  sync.Mutex,
	buffer_lock: sync.Mutex,

	base:      []u32, // baked plate + rim + shadow
	cover:     []u8,  // baked plate coverage, for the live glow pass
	dist:      []f32, // baked distance from the plate centre
	timer_on:  bool,
	visible:   bool,

	phase:     f32, // braid travel
	reveal:    f32, // 0 hidden .. 1 shown
	power:     [COLUMNS * ROWS]f32,
	fresh:     [COLUMNS * ROWS]f32,
	cell_mask: [COLUMNS * ROWS]f32, // baked plate mask per dot, scale-bound
	mask_scale: f32,                // hud.scale the mask was baked for
	buffer:    ^[dynamic]f32,
	last:      time.Time,
	ready:     time.Time,
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

SUPERSAMPLE :: 2
// Column block of the splat coverage table: columns are tabulated once per
// block so the inner loop only adds and compares.
SPLAT_BLOCK :: 64

hud_setup_dib :: proc() {
	hud.w = int(f32(WINDOW_W) * hud.scale)
	hud.h = int(f32(WINDOW_H) * hud.scale)
	hud.hi = make([]u32, hud.w * SUPERSAMPLE * hud.h * SUPERSAMPLE)

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

	n := hud.w * SUPERSAMPLE * hud.h * SUPERSAMPLE
	hud.base = make([]u32, n)
	hud.cover = make([]u8, n)
	hud.dist = make([]f32, n)
	hud_build_base()
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
		sync.unlock(&hud.buffer_lock)
		capture_discard()
		hud_set_pump(true)
		hud.recording = true
		hud.reveal = 1
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
		capture_pump(nil, hud.buffer, nil) // sweep the final period
		sync.unlock(&hud.buffer_lock)
		capture_stop(nil)
		hud.capture_up = false
		log_write("[hex] mic released, transcribing")
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
				if hud.buffer != nil { capture_pump(nil, hud.buffer, nil) }
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
		log_write("[hex] cancelled recording")
	} else if hud_is_busy() {
		sync.lock(&hud.state_lock)
		hud.cancel_gen += 1
		sync.unlock(&hud.state_lock)
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
SEG_POLL_MS :: 250
SEG_SILENCE_MS :: 300    // silence that closes a segment

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
		defer delete(full)
	}
	if paste_text(full) {
		log_write("[hex] pasted %.2fs: %s", elapsed, full)
	} else {
		log_write("[hex] paste failed")
	}
}

hud_tick :: proc() {
	now := time.now()
	dt := f32(time.duration_seconds(time.since(hud.last)))
	if dt > 0.1 { dt = 0.1 }
	hud.last = now

	// Level: fast attack, slow release, like the scope's reservoir.
	target := f32(0)
	if hud.recording {
		sync.lock(&hud.buffer_lock)
		if hud.buffer != nil && len(hud.buffer^) > 0 {
			n := min(len(hud.buffer^), 2048)
			sum: f64
			for i in 0 ..< n {
				v := hud.buffer^[len(hud.buffer^) - 1 - i]
				sum += f64(v) * f64(v)
			}
			rms := f32(math.sqrt(sum / f64(n)))
			target = clamp((rms - 0.002) * 22.0, 0, 1)
			// A quiet room still breathes: Hex's Auto level follows recent peak.
			idle := 0.10 + 0.05 * (0.5 + 0.5 * math.sin_f32(hud.phase * 0.05))
			target = max(target, idle)
		}
		sync.unlock(&hud.buffer_lock)
	}
	busy := hud_is_busy()
	if os.get_env("HEX_DEMO", context.temp_allocator) != "" && !hud.recording {
		target = 0.35 + 0.3 * (0.5 + 0.5 * math.sin_f32(f32(time.duration_seconds(time.since(hud.last))) * 3.0 + hud.phase * 0.06))
	}
	if target > hud.level {
		hud.level += (target - hud.level) * min(1, dt * 22)
	} else {
		hud.level += (target - hud.level) * min(1, dt * 6)
	}
	want_cool: f32 = busy ? 1 : 0
	hud.cool += (want_cool - hud.cool) * min(1, dt * (want_cool > hud.cool ? 6 : 3))
	if hud.level > hud.peak { hud.peak = hud.level }
	hud.peak = max(0, hud.peak - dt * 1.6)

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

	hud_advance_dots(dt)
	hud_render()
	hud_resolve()
	hud_blit()
}

// One frame of the braid: two travelling traces offset by half a cycle, each
// modulated across rows by a helix, decaying like phosphor. This is the CPU
// stand-in for dot_scope.rs's Voice simulation.
hud_advance_dots :: proc(dt: f32) {
	busy := hud_is_busy()
	hud.phase += dt * 26.0
	sigma: f32 = 2.4
	decay := math.exp_f32(-dt * 6.5)

	// The column envelope only depends on the head, the helix only on the row,
	// so both are built once per frame instead of once per cell.
	env: [2][COLUMNS]f32
	for k in 0 ..< 2 {
		head := hud.phase + f32(k) * (f32(COLUMNS) * 0.5)
		head_mod := math.mod(head, f32(COLUMNS))
		for i in 0 ..< COLUMNS {
			d := math.mod(f32(i) - head_mod + f32(COLUMNS) * 1.5, f32(COLUMNS)) - f32(COLUMNS) * 0.5
			env[k][i] = math.exp_f32(-(d * d) / (2 * sigma * sigma))
		}
	}
	helix: [2][ROWS]f32
	for k in 0 ..< 2 {
		for j in 0 ..< ROWS {
			helix[k][j] = 0.5 + 0.5 * f32(math.sin(f64((f32(j) - f32(ROWS-1)*0.5) * 1.6 * math.PI / f32(ROWS) + hud.phase * 0.05 + f32(k) * math.PI)))
		}
	}
	energy := hud.level * 1.35
	if busy { energy = max(energy, 0.18) }

	for i in 0 ..< COLUMNS {
		for j in 0 ..< ROWS {
			idx := j * COLUMNS + i
			excite: f32 = 0
			for k in 0 ..< 2 {
				excite = max(excite, env[k][i] * helix[k][j])
			}
			v := excite * energy
			if v > 0.02 {
				hud.fresh[idx] = 1
			}
			decayed := hud.power[idx] * decay
			hud.power[idx] = clamp(max(decayed, v), 0, 1)
			hud.fresh[idx] = max(0, hud.fresh[idx] - dt * 2.2)
		}
	}
}

// --- drawing ---------------------------------------------------------------

plate_center :: proc() -> (x, y: f32) {
	// Window centre, with the plate a touch above to leave room for its shadow.
	return f32(hud.w * SUPERSAMPLE) * 0.5, f32(hud.h * SUPERSAMPLE) * 0.5 - 3.0 * hud.scale * f32(SUPERSAMPLE)
}

rounded_box :: proc(px, py, hx, hy, r: f32) -> f32 {
	qx := abs(px) - hx + r
	qy := abs(py) - hy + r
	return math.sqrt_f32(max(qx, 0) * max(qx, 0) + max(qy, 0) * max(qy, 0)) + min(max(qx, qy), 0) - r
}

sstep :: proc(e0, e1, x: f32) -> f32 {
	t := clamp((x - e0) / (e1 - e0), 0, 1)
	return t * t * (3 - 2 * t)
}

coverage :: proc(d, softness: f32) -> f32 {
	return sstep(softness, -softness, d)
}

mix :: proc(a, b, t: f32) -> f32 {
	return a + (b - a) * t
}

mix3 :: proc(a, b: Vec3, t: f32) -> Vec3 {
	return Vec3{a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t}
}

// Palette from dot_scope::palette(dark=true, cool=0, fail=0).
PAL_YOU :: Vec3{0.86, 0.83, 0.77}
PAL_YOU_CORE :: Vec3{0.98, 0.96, 0.92}
PAL_AGENT :: Vec3{0.70, 0.78, 0.90}
PAL_AGENT_CORE :: Vec3{0.92, 0.95, 1.0}
PAL_PLATE :: Vec3{17.0 / 255.0, 17.0 / 255.0, 20.0 / 255.0}
PAL_PLATE_A :: 0.94
PAL_RIM_A :: 0.11
PAL_SHADOW_A :: 0.5

// Baked once: plate, rim and drop shadow never change while the HUD runs, so
// the per-frame path only copies them and adds the two dynamic layers (glow,
// dots). Also caches coverage and radius so the glow pass has no rounded_box
// or sqrt in it.
hud_build_base :: proc() {
	px_w := hud.w * SUPERSAMPLE
	px_h := hud.h * SUPERSAMPLE

	cx, cy := plate_center()
	scale := hud.scale * f32(SUPERSAMPLE)
	hx := PLATE_W * 0.5 * scale
	hy := PLATE_H * 0.5 * scale
	radius := PLATE_RADIUS * scale
	aa := 0.75 * scale

	for y in 0 ..< px_h {
		for x in 0 ..< px_w {
			px := f32(x) + 0.5 - cx
			py := f32(y) + 0.5 - cy
			i := y*px_w + x

			hud.dist[i] = math.sqrt_f32(px*px + py*py)
			d := rounded_box(px, py, hx, hy, radius)
			shape := coverage(d, aa)
			hud.cover[i] = u8(clamp(shape * 255, 0, 255))
			if shape <= 0 {
				hud.base[i] = 0
				continue
			}

			// shadow: same box shifted 3 pt down, blurred by tanh
			sd := rounded_box(px, py - 3 * scale, hx, hy, radius)
			shadow := PAL_SHADOW_A * (0.5 - 0.5 * f32(math.tanh(f64(sd / (6.5 * scale)))))

			r := PAL_PLATE[0] * PAL_PLATE_A * shape
			g := PAL_PLATE[1] * PAL_PLATE_A * shape
			b := PAL_PLATE[2] * PAL_PLATE_A * shape
			a := PAL_PLATE_A * shape

			// drop shadow behind the plate
			sr: f32 = 0
			sg: f32 = 0
			sb: f32 = 0
			sa := shadow * (1 - shape) * 0.9
			if sa > 0 {
				r = sr * sa + r * (1 - sa)
				g = sg * sa + g * (1 - sa)
				b = sb * sa + b * (1 - sa)
				a = sa + a * (1 - sa)
			}

			// rim: a 1 px lighter edge
			rim := PAL_RIM_A * coverage(abs(d) - 0.3 * scale, aa)
			if rim > 0 {
				r = 1.0 * rim + r * (1 - rim)
				g = 1.0 * rim + g * (1 - rim)
				b = 1.0 * rim + b * (1 - rim)
				a = rim + a * (1 - rim)
			}

			hud.base[i] = pack(r, g, b, a)
		}
	}
}

// The plate mask depends only on the dot grid and the plate scale, never on the
// live level, so it is baked once per scale instead of re-running rounded_box
// and its smoothstep for every dot on every frame. Same expressions, same
// order, so the baked values are the ones the loop used to compute.
hud_bake_cell_mask :: proc() {
	scale := hud.scale * f32(SUPERSAMPLE)
	hx := PLATE_W * 0.5 * scale
	hy := PLATE_H * 0.5 * scale
	radius := PLATE_RADIUS * scale
	mask_lo := 1.4 * scale
	mask_hi := 4.6 * scale
	center_off_x := (f32(COLUMNS) - 1) * 0.5
	center_off_y := (f32(ROWS) - 1) * 0.5
	for j in 0 ..< ROWS {
		dy := (f32(j) - center_off_y) * PITCH * scale
		for i in 0 ..< COLUMNS {
			dx := (f32(i) - center_off_x) * PITCH * scale
			cap := -rounded_box(dx, dy, hx, hy, radius)
			hud.cell_mask[j * COLUMNS + i] = sstep(mask_lo, mask_hi, cap)
		}
	}
	hud.mask_scale = hud.scale
}

hud_render :: proc() {
	ensure_unpack_lut()
	if hud.mask_scale != hud.scale { hud_bake_cell_mask() }
	copy(hud.hi, hud.base)
	hi := hud.hi

	cx, cy := plate_center()
	scale := hud.scale * f32(SUPERSAMPLE)
	hx := PLATE_W * 0.5 * scale
	hy := PLATE_H * 0.5 * scale
	aa := 0.75 * scale

	px_w := hud.w * SUPERSAMPLE
	px_h := hud.h * SUPERSAMPLE

	// Centre glow, visible while listening. Uses the baked coverage and
	// distance; only the palette mix is live.
	glow := max(hud.level, hud.peak * 0.6) * hud.reveal
	if glow > 0.01 {
		reach := (34.0 + 20.0 * glow) * scale
		strength := 0.19 * 0.55 * min(1, glow)
		middle := mix3(mix3(PAL_YOU, PAL_AGENT, hud.cool), mix3(PAL_YOU_CORE, PAL_AGENT_CORE, hud.cool), 0.5)
		base_col := mix3(PAL_YOU, PAL_AGENT, hud.cool)
		x0 := max(0, int(cx - hx - 2 * scale))
		x1 := min(px_w - 1, int(cx + hx + 2 * scale))
		y0 := max(0, int(cy - hy - 2 * scale))
		y1 := min(px_h - 1, int(cy + hy + 2 * scale))
		// The mix endpoint table is per frame, not per pixel.
		dm := Vec3{base_col[0] - middle[0], base_col[1] - middle[1], base_col[2] - middle[2]}
		// The baked coverage is a rounded box and the baked distance is a disc,
		// so on each row the pixels the glow touches are one contiguous run:
		// find its ends by scanning in from the plate box, then paint the run
		// with no tests inside. Every pixel in the run has coverage > 0,
		// distance < reach and weight > 0, so nothing that used to be skipped
		// is painted.
		for y in y0 ..= y1 {
			row := y * px_w
			lo := x0
			hi_x := x1
			for lo <= hi_x && (hud.cover[row + lo] == 0 || hud.dist[row + lo] >= reach) { lo += 1 }
			for hi_x >= lo && (hud.cover[row + hi_x] == 0 || hud.dist[row + hi_x] >= reach) { hi_x -= 1 }
			for x in lo ..= hi_x {
				i := row + x
				shape := unpack_channel(u32(hud.cover[i]))
				d := hud.dist[i]
				t := d / reach
				ga := strength * (1 - t) * shape
				gc0 := middle[0] + dm[0] * t
				gc1 := middle[1] + dm[1] * t
				gc2 := middle[2] + dm[2] * t
				ng := 1 - ga
				r, g, b, a := unpack(hi[i])
				hi[i] = pack_unsaturated(gc0*ga + r*ng, gc1*ga + g*ng, gc2*ga + b*ng, ga + a*ng)
			}
		}
	}

	// Dots.
	center_off_x := (f32(COLUMNS) - 1) * 0.5
	center_off_y := (f32(ROWS) - 1) * 0.5
	presence := 0.5 + 0.5 * max(hud.level, hud.reveal)
	soft := 0.7 * hud.scale
	base_col := mix3(PAL_YOU, PAL_AGENT, hud.cool)
	core_col := mix3(PAL_YOU_CORE, PAL_AGENT_CORE, hud.cool)
	for j in 0 ..< ROWS {
		// dy, the row envelope and the floor alpha depend on the row only.
		dy := (f32(j) - center_off_y) * PITCH * scale
		row := abs(dy) / hy
		floor_alpha := 0.10 * (1 - 0.7 * sstep(0.35, 0.95, row)) * presence * hud.reveal
		for i in 0 ..< COLUMNS {
			power := hud.power[j * COLUMNS + i]
			fresh := hud.fresh[j * COLUMNS + i]

			dx := (f32(i) - center_off_x) * PITCH * scale
			mask := hud.cell_mask[j * COLUMNS + i]
			if mask <= 0 { continue }

			curve: f32 = 0.75
			mix_t := curve * (1.4 + (1.0 - 1.4) * fresh)
			alpha := floor_alpha + (1 - floor_alpha) * f32(math.pow(f64(power), f64(mix_t)))
			dot_radius := (0.27 + 0.17 * f32(math.pow(f64(power), 0.6))) * (1.0 + (mix(0.82, 1.0, fresh) - 1.0) * sstep(0.03, 0.2, power)) * PITCH * scale
			core := clamp(power * power * 1.1, 0, 0.9) * (0.3 + 0.7 * fresh)
			col := mix3(base_col, core_col, core)

			sx := cx + dx
			sy := cy + dy
			rad := dot_radius * 2.4

			// Halo (the shader's wide, faint pass) then the dot body.
			if power > 0.05 {
				halo := 0.132 * power * power * mask * coverage(rad * 0.85, aa)
				splat(hi, px_w, px_h, sx, sy, rad, col, halo, soft)
			}
			splat(hi, px_w, px_h, sx, sy, dot_radius, col, alpha * mask, soft)
		}
	}
}

splat :: proc(buf: []u32, bw, bh: int, cx, cy, radius: f32, col: Vec3, alpha, soft: f32) {
	if alpha <= 0.001 { return }
	x0 := max(0, int(cx - radius - 1))
	x1 := min(bw - 1, int(cx + radius + 1))
	y0 := max(0, int(cy - radius - 1))
	y1 := min(bh - 1, int(cy + radius + 1))
	// coverage() is zero from d == radius + soft outwards, so the square corners
	// are rejected on squared distance: no sqrt, no divide, and no pixel that
	// would have been painted is skipped.
	r_out := radius + soft
	r_out2 := r_out * r_out
	// Inside radius - soft the smoothstep argument is already past 1, so the
	// coverage is exactly 1: those pixels take the plain alpha and skip the
	// sqrt and the divide. r_in2 stays negative when the dot is too small to
	// have a full-coverage core, so the test then never matches.
	r_in := radius - soft
	r_in2: f32 = -1
	if r_in > 0 { r_in2 = r_in * r_in }
	// dx and dx*dx are the same for every row, so they are tabulated once per
	// column block instead of being recomputed per pixel. Each pixel is still
	// blended exactly once per call, so the result is unchanged; the block is
	// only there so a huge radius cannot overflow the table.
	dx2s: [SPLAT_BLOCK]f32
	for xb := x0; xb <= x1; xb += SPLAT_BLOCK {
		xe := min(x1, xb + SPLAT_BLOCK - 1)
		for x in xb ..= xe {
			dx := f32(x) + 0.5 - cx
			dx2s[x - xb] = dx * dx
		}
		for y in y0 ..= y1 {
			dy := f32(y) + 0.5 - cy
			dy2 := dy * dy
			for x in xb ..= xe {
				dd := dx2s[x - xb] + dy2
				if dd >= r_out2 { continue }
				a: f32
				if dd <= r_in2 {
					a = alpha
				} else {
					cov := coverage(math.sqrt_f32(dd) - radius, soft)
					if cov <= 0 { continue }
					a = alpha * cov
				}
				na := 1 - a
				idx := y*bw + x
				dr, dg, db, da := unpack(buf[idx])
				r := col[0] * a + dr * na
				g := col[1] * a + dg * na
				b := col[2] * a + db * na
				al := a + da * na
				buf[idx] = pack_unsaturated(r, g, b, al)
			}
		}
	}
}

// Premultiplied 8-bit per channel, BGRA (little-endian DIB word).
pack :: proc(r, g, b, a: f32) -> u32 {
	ra := u32(clamp(r * 255, 0, 255))
	ga := u32(clamp(g * 255, 0, 255))
	ba := u32(clamp(b * 255, 0, 255))
	aa := u32(clamp(a * 255, 0, 255))
	return ba | (ga << 8) | (ra << 16) | (aa << 24)
}

// Blend loops write premultiplied channels that are already convex
// combinations of [0,1] values, so the clamp in pack() cannot fire there:
// this drops the four min/max pairs. The bit-exact render check proves it.
pack_unsaturated :: proc(r, g, b, a: f32) -> u32 {
	ra := u32(r * 255)
	ga := u32(g * 255)
	ba := u32(b * 255)
	aa := u32(a * 255)
	return ba | (ga << 8) | (ra << 16) | (aa << 24)
}

// 8-bit -> [0,1] conversion dominates the blend loops; the table holds the very
// same f32(x)/255 values and removes four divides per pixel.
UNPACK_LUT: [256]f32
unpack_lut_ready: bool

ensure_unpack_lut :: proc() {
	if unpack_lut_ready { return }
	for i in 0 ..< 256 { UNPACK_LUT[i] = f32(i) / 255.0 }
	unpack_lut_ready = true
}

unpack_channel :: proc(v: u32) -> f32 {
	return UNPACK_LUT[v & 0xff]
}

unpack :: proc(v: u32) -> (r, g, b, a: f32) {
	ensure_unpack_lut()
	return unpack_channel(v >> 16), unpack_channel(v >> 8), unpack_channel(v), unpack_channel(v >> 24)
}

// Box-downsample the supersampled frame into the DIB (premultiplied, so a
// plain average is correct).
hud_resolve :: proc() {
	ss := SUPERSAMPLE
	area := u32(ss * ss)
	scale := u32(hud.reveal / f32(area) * 65536.0)
	stride := hud.w * ss
	// The DIB and the supersampled field are sized together by hud_init; bail
	// out rather than index past either if a resize ever leaves them apart.
	if len(hud.hi) < stride * (hud.h - 1) * ss + stride ||
	   len(hud.pixels) < hud.w * hud.h {
		return
	}
	// Hoist the slice headers and extents: with them reloaded from the global
	// every pixel the loop cannot be vectorized, which doubled this pass.
	hi := hud.hi
	px := hud.pixels
	w := hud.w
	h := hud.h
	#no_bounds_check {
		for y in 0 ..< h {
			out := y * w
			row0 := y * ss * stride
			for x in 0 ..< w {
				// Two 16-bit lanes per word hold the b,g pair and the r,a pair,
				// so four shifts and four adds per sample instead of four
				// shift/mask pairs and four adds per channel. Each lane sums at
				// most ss*ss*255, which cannot overflow 16 bits for any sane
				// supersample factor, so the totals are the plain byte sums.
				lo, hivec: u32
				i := row0 + x * ss
				for sy in 0 ..< ss {
					row := i + sy * stride
					for sx in 0 ..< ss {
						v := hi[row + sx]
						lo += v & 0x00FF00FF
						hivec += (v >> 8) & 0x00FF00FF
					}
				}
				bs := lo & 0xffff
				rs := lo >> 16
				gs := hivec & 0xffff
				as := hivec >> 16
				px[out + x] = ((bs * scale) >> 16) |
					(((gs * scale) >> 16) << 8) |
					(((rs * scale) >> 16) << 16) |
					(((as * scale) >> 16) << 24)
			}
		}
	}
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

