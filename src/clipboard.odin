package main

import "core:strings"
import "core:sys/windows"

// Clipboard primitives for paste-style insertion. Windows hands the clipboard
// to one process at a time, so every call retries briefly instead of failing:
// Office, RDP and clipboard managers hold it for tens of milliseconds.

CLIP_RETRIES :: 12

clip_sequence :: proc() -> u32 {
	return windows.GetClipboardSequenceNumber()
}

clip_open :: proc() -> bool {
	for _ in 0 ..< CLIP_RETRIES {
		if windows.OpenClipboard(nil) != windows.BOOL(false) { return true }
		windows.Sleep(10)
	}
	return false
}

// Returns a heap copy of the clipboard text. `ok` is false when the clipboard
// holds no text (files, image, HTML-only) or cannot be opened.
clip_read_text :: proc() -> (text: string, ok: bool) {
	if !clip_open() { return "", false }
	defer windows.CloseClipboard()

	if windows.IsClipboardFormatAvailable(windows.CF_UNICODETEXT) == windows.BOOL(false) {
		return "", false
	}
	h := windows.GetClipboardData(windows.CF_UNICODETEXT)
	if h == nil { return "", false }
	locked := windows.GlobalLock(windows.HGLOBAL(h))
	if locked == nil { return "", false }
	defer windows.GlobalUnlock(windows.HGLOBAL(h))

	chars := cast([^]u16)locked
	n := 0
	for chars[n] != 0 { n += 1 }
	s, err := windows.utf16_to_utf8(chars[:n], context.temp_allocator)
	if err != nil { return "", false }
	return strings.clone(s), true
}

clip_write_text :: proc(text: string) -> bool {
	if !clip_open() { return false }
	defer windows.CloseClipboard()

	if windows.EmptyClipboard() == windows.BOOL(false) { return false }

	utf16 := windows.utf8_to_utf16(text, context.temp_allocator)
	size := (len(utf16) + 1) * size_of(u16)
	h := windows.GlobalAlloc(windows.GMEM_MOVEABLE | windows.GMEM_ZEROINIT, windows.SIZE_T(size))
	if h == nil { return false }
	locked := windows.GlobalLock(windows.HGLOBAL(h))
	if locked == nil {
		windows.GlobalFree(h)
		return false
	}
	dst := cast([^]u16)locked
	for ch, i in utf16 { dst[i] = ch }
	dst[len(utf16)] = 0
	windows.GlobalUnlock(windows.HGLOBAL(h))

	if windows.SetClipboardData(windows.CF_UNICODETEXT, windows.HANDLE(h)) == nil {
		windows.GlobalFree(h)
		return false
	}
	return true
}

// Formats that Windows, OLE and clipboard managers attach next to the payload.
// They carry no user content and Hex can neither produce nor restore them, so a
// clipboard made of text plus these is still safe to replace. Every one of
// them shows up on an ordinary copy from Office, Edge or Explorer.
METADATA_FORMATS := []string{
	"DataObject",
	"Ole Private Data",
	"Object Descriptor",
	"Link Source",
	"Link Source Descriptor",
	"DataObjectAttributes",
	"EnterpriseDataProtectionId",
	"CanIncludeInClipboardHistory",
	"CanUploadToCloudClipboard",
	"ExcludeClipboardContentFromMonitorProcessing",
	"Clipboard Viewer Ignore",
}

clip_format_is_metadata :: proc(format: windows.UINT) -> bool {
	buf: [128]u16
	n := windows.GetClipboardFormatNameW(format, &buf[0], i32(len(buf)))
	if n <= 0 { return false }
	name, err := windows.utf16_to_utf8(buf[:n], context.temp_allocator)
	if err != nil { return false }
	for meta in METADATA_FORMATS {
		if strings.equal_fold(name, meta) { return true }
	}
	return false
}

// Human-readable name for a clipboard format, for the log.
clip_format_display_name :: proc(format: windows.UINT) -> string {
	buf: [128]u16
	n := windows.GetClipboardFormatNameW(format, &buf[0], i32(len(buf)))
	if n > 0 {
		name, err := windows.utf16_to_utf8(buf[:n], context.temp_allocator)
		if err == nil { return name }
	}
	switch format {
	case windows.CF_BITMAP:       return "CF_BITMAP"
	case windows.CF_METAFILEPICT: return "CF_METAFILEPICT"
	case windows.CF_DIB:          return "CF_DIB"
	case windows.CF_ENHMETAFILE:  return "CF_ENHMETAFILE"
	case windows.CF_HDROP:        return "CF_HDROP"
	case windows.CF_DIBV5:        return "CF_DIBV5"
	}
	return "registered format"
}

// The formats that make the clipboard non-restorable, comma separated, for the
// log line that explains why a transcript replaced them.
clip_content_format_names :: proc() -> string {
	if !clip_open() { return "unreadable" }
	defer windows.CloseClipboard()
	b: strings.Builder
	strings.builder_init(&b, context.temp_allocator)
	format: windows.UINT
	for {
		format = windows.EnumClipboardFormats(format)
		if format == 0 { break }
		if format == windows.CF_UNICODETEXT || format == windows.CF_TEXT ||
			format == windows.CF_OEMTEXT || format == windows.CF_LOCALE { continue }
		if clip_format_is_metadata(format) { continue }
		if strings.builder_len(b) > 0 { strings.write_string(&b, ", ") }
		strings.write_string(&b, clip_format_display_name(format))
	}
	return strings.to_string(b)
}

// True when the clipboard holds something Hex cannot put back after a paste:
// images, file lists, rich text, metafiles. Text plus bookkeeping formats is
// replaceable; any other format is content the user would lose.
clip_has_nontext :: proc() -> bool {
	if !clip_open() { return false }
	defer windows.CloseClipboard()
	format: windows.UINT
	has_unicode := false
	has_ansi := false
	has_content := false
	for {
		format = windows.EnumClipboardFormats(format)
		if format == 0 { break }
		switch format {
		case windows.CF_UNICODETEXT:
			has_unicode = true
		case windows.CF_TEXT, windows.CF_OEMTEXT:
			has_ansi = true
		case windows.CF_LOCALE:
			// Metadata; not content.
		case:
			if !clip_format_is_metadata(format) { has_content = true }
		}
	}
	if has_content { return true }
	// CF_TEXT is often a compatibility twin of CF_UNICODETEXT. Allow it only
	// when the Unicode text that Hex can restore is also present.
	return has_ansi && !has_unicode
}

key_down :: proc(vk: i32) -> bool {
	return windows.GetAsyncKeyState(vk) < 0
}

// Ctrl+Space is still held when the transcript is ready; Shift or Alt held at
// that moment would turn Ctrl+V into Ctrl+Shift+V / Ctrl+Alt+V in the target.
wait_conflicting_modifiers :: proc() {
	for _ in 0 ..< 300 { // 3 s ceiling
		if !key_down(i32(windows.VK_SHIFT)) &&
		   !key_down(i32(windows.VK_MENU)) &&
		   !key_down(i32(windows.VK_LWIN)) &&
		   !key_down(i32(windows.VK_RWIN)) {
			return
		}
		windows.Sleep(10)
	}
}
