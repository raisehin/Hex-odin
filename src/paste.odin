package main

import "core:fmt"
import "core:strings"
import "core:sys/windows"
import "core:thread"
import "core:time"

KEYEVENTF_KEYUP :: 0x0002
KEYEVENTF_UNICODE :: 0x0004

// How the transcript reaches the focused window. Hex pastes: it puts the text
// on the clipboard, sends the paste chord, and puts the old clipboard back half
// a second later. Typing is kept only as an explicit fallback for apps that
// reject synthetic paste.
PasteMode :: enum {
	CtrlV,
	CtrlShiftV,
	ShiftInsert,
	Type,
}

CLIPBOARD_RESTORE_DELAY :: 500 * time.Millisecond

key_input :: proc(vk: u16, up: bool) -> windows.INPUT {
	ev: windows.INPUT
	ev.type = .KEYBOARD
	ev.ki.wVk = vk
	if up { ev.ki.dwFlags = KEYEVENTF_KEYUP }
	return ev
}

send_paste_chord :: proc(mode: PasteMode) -> bool {
	if mode == .Type { return false }
	// A physically held Ctrl stays held: synthesizing its release would leave
	// Windows believing the key is up while the user still presses it.
	ctrl_held := key_down(i32(windows.VK_CONTROL))

	inputs: [6]windows.INPUT
	n := 0
	need_ctrl := mode != .ShiftInsert
	if need_ctrl && !ctrl_held {
		inputs[n] = key_input(u16(windows.VK_CONTROL), false); n += 1
	}
	switch mode {
	case .CtrlShiftV:
		inputs[n] = key_input(u16(windows.VK_SHIFT), false); n += 1
		inputs[n] = key_input(u16(windows.VK_V), false); n += 1
		inputs[n] = key_input(u16(windows.VK_V), true); n += 1
		inputs[n] = key_input(u16(windows.VK_SHIFT), true); n += 1
	case .ShiftInsert:
		ctrl_held = false // Shift+Insert leaves Ctrl alone entirely
		inputs[n] = key_input(u16(windows.VK_SHIFT), false); n += 1
		inputs[n] = key_input(u16(windows.VK_INSERT), false); n += 1
		inputs[n] = key_input(u16(windows.VK_INSERT), true); n += 1
		inputs[n] = key_input(u16(windows.VK_SHIFT), true); n += 1
	case .CtrlV, .Type:
		inputs[n] = key_input(u16(windows.VK_V), false); n += 1
		inputs[n] = key_input(u16(windows.VK_V), true); n += 1
	}
	if need_ctrl && !ctrl_held {
		inputs[n] = key_input(u16(windows.VK_CONTROL), true); n += 1
	}
	sent := windows.SendInput(u32(n), &inputs[0], i32(size_of(windows.INPUT)))
	return int(sent) == n
}

Restore_Job :: struct {
	text:    string, // original clipboard text (owned)
	had:     bool,   // original clipboard held text at all
	seq:     u32,    // sequence right after our transcript was placed
}

restore_worker :: proc(t: ^thread.Thread) {
	job := cast(^Restore_Job)t.data
	time.sleep(CLIPBOARD_RESTORE_DELAY)
	defer {
		if len(job.text) > 0 { delete(job.text) }
		free(job)
	}
	// Someone copied something after us: their clipboard wins.
	if clip_sequence() != job.seq { return }
	if job.had {
		clip_write_text(job.text)
	}
}

type_text :: proc(text: string) -> bool {
	if len(text) == 0 { return true }
	utf16 := windows.utf8_to_utf16(text, context.temp_allocator)
	inputs := make([]windows.INPUT, len(utf16) * 2, context.temp_allocator)
	for ch, i in utf16 {
		down: windows.INPUT
		down.type = .KEYBOARD
		down.ki.wScan = ch
		down.ki.dwFlags = KEYEVENTF_UNICODE

		up: windows.INPUT
		up.type = .KEYBOARD
		up.ki.wScan = ch
		up.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP

		inputs[i*2] = down
		inputs[i*2+1] = up
	}
	// SendInput caps how many events one call accepts; a long transcript is fed
	// in slices so nothing is silently dropped.
	sent_total := 0
	for start := 0; start < len(inputs); start += 256 {
		end := min(start + 256, len(inputs))
		sent := windows.SendInput(u32(end - start), &inputs[start], i32(size_of(windows.INPUT)))
		sent_total += int(sent)
		if int(sent) != end - start { break }
	}
	return sent_total == len(inputs)
}

paste_text :: proc(text: string) -> bool {
	if len(strings.trim_space(text)) == 0 {
		log_write("[hex] refusing to paste an empty transcript")
		return false
	}
	cfg := config_load()
	mode: PasteMode
	switch cfg.paste_keys {
	case "type": mode = .Type
	case "shift_insert": mode = .ShiftInsert
	case "ctrl_shift_v": mode = .CtrlShiftV
	case: mode = .CtrlV
	}
	if mode == .Type { return type_text(text) }

	wait_conflicting_modifiers()

	before := clip_sequence()
	original, had_text := clip_read_text()
	if clip_sequence() != before {
		// The user copied something while the transcript was being prepared.
		if had_text { delete(original) }
		original, had_text = clip_read_text()
	}
	if !had_text { original = "" }
	if clip_has_nontext() {
		// An image, file list or rich text is on the clipboard. Pasting is what
		// the user asked for, so the transcript replaces it; only the text
		// clipboard is restored, so say which payload was dropped.
		log_write(fmt.aprintf("[hex] replacing non-restorable clipboard (%s) with the transcript",
			clip_content_format_names()))
	}

	if !clip_write_text(text) {
		log_write("[hex] could not open the clipboard; nothing pasted")
		if had_text { delete(original) }
		return false
	}
	after := clip_sequence()

	if !send_paste_chord(mode) {
		log_write("[hex] SendInput rejected the paste chord (elevated target?)")
		if clip_sequence() == after { clip_write_text(original) }
		if had_text { delete(original) }
		return false
	}

	if had_text {
		job := new(Restore_Job)
		if job == nil {
			delete(original)
			log_write("[hex] out of memory scheduling clipboard restore")
			return true
		}
		job^ = Restore_Job{text = original, had = true, seq = after}
		t := thread.create(restore_worker)
		if t == nil {
			delete(original)
			free(job)
			log_write("[hex] could not create clipboard restore thread")
			return true
		}
		t.data = job
		t.flags += {.Self_Cleanup}
		thread.start(t)
	} else {
		log_write("[hex] clipboard held no text; transcript left there")
	}
	return true
}

