package main

import "core:fmt"
import "core:os"
import "core:strings"
import "core:sync"
import "core:time"
import "core:sys/windows"

// The app hides its console, so anything worth knowing has to land in a file.
// One append-only log under the shared data root; `hex app` mirrors it to the
// terminal.
// The path and the data dirs are resolved once: log_write runs on the UI thread
// for every event, and rebuilding the path through context.temp_allocator grew
// the temp arena on every line. The handle is opened once and kept for the
// process lifetime: open/append/close per line cost ~250 us of syscalls, which
// is latency on the hotkey path.
//
// Lines are also batched: a WriteFile per line is ~8 us of kernel time on the
// hotkey path, so lines accumulate in a fixed 8 KiB buffer and go out when it
// fills, when 500 ms have passed, or when log_flush is called (every dictation
// finishes with one, so nothing a user would read after an ordinary run is
// lost). Only a hard crash can drop the last unflushed lines.
log_lock: sync.Mutex
log_path: string
log_file: ^os.File
log_ready: bool
BACKLOG :: 8 * 1024
log_backlog: [BACKLOG]u8
log_backlog_len: int
log_pending_since: time.Time
// The terminal mirror is a write to stdout per line (~1.5 us). It only makes
// sense when a terminal is actually attached: the resident app hides its
// console, so the write would go nowhere and still cost the hotkey path.
log_mirror: bool
log_mirror_known: bool
// Reused across lines so formatting a log line does not allocate at all.
log_msg_buf: strings.Builder
log_line_buf: strings.Builder

// "[YYYY-MM-DD HH:MM:SS] " with the same zero padding the format string used.
log_write_stamp :: proc(b: ^strings.Builder, st: windows.SYSTEMTIME) {
	strings.write_byte(b, '[')
	log_write_u32(b, u32(st.year), 4)
	strings.write_byte(b, '-')
	log_write_u32(b, u32(st.month), 2)
	strings.write_byte(b, '-')
	log_write_u32(b, u32(st.day), 2)
	strings.write_byte(b, ' ')
	log_write_u32(b, u32(st.hour), 2)
	strings.write_byte(b, ':')
	log_write_u32(b, u32(st.minute), 2)
	strings.write_byte(b, ':')
	log_write_u32(b, u32(st.second), 2)
	strings.write_string(b, "] ")
}

// Zero-padded decimal, at least `width` digits, no allocation.
log_write_u32 :: proc(b: ^strings.Builder, v: u32, width: int) {
	digits: [10]u8
	n := 0
	x := v
	for {
		digits[n] = u8('0' + x % 10)
		n += 1
		x /= 10
		if x == 0 { break }
	}
	for n < width {
		digits[n] = '0'
		n += 1
	}
	for n > 0 {
		n -= 1
		strings.write_byte(b, digits[n])
	}
}

log_flush_unlocked :: proc() {
	if log_backlog_len == 0 { return }
	if log_file == nil {
		if f, err := os.open(log_path, {.Append, .Create}); err == nil { log_file = f }
	}
	if log_file != nil {
		if _, err := os.write(log_file, log_backlog[:log_backlog_len]); err != nil {
			os.close(log_file)
			log_file = nil
		}
	}
	log_backlog_len = 0
	log_pending_since = {}
}

log_flush :: proc() {
	sync.lock(&log_lock)
	log_flush_unlocked()
	sync.unlock(&log_lock)
}

log_write :: proc(format: string, args: ..any) {
	sync.lock(&log_lock)
	defer sync.unlock(&log_lock)

	if !log_ready {
		cfg := config_load()
		os.make_directory(cfg.data_dir)
		os.make_directory(cfg.logs_dir)
		log_path = strings.concatenate({cfg.logs_dir, "/odin.log"})
		log_ready = true
	}
	if !log_mirror_known {
		log_mirror_known = true
		h := windows.GetStdHandle(windows.STD_OUTPUT_HANDLE)
		mode: u32
		log_mirror = h != windows.INVALID_HANDLE_VALUE && bool(windows.GetConsoleMode(h, &mode))
	}

	strings.builder_reset(&log_msg_buf)
	if len(args) == 0 {
		strings.write_string(&log_msg_buf, format)
	} else {
		fmt.sbprintf(&log_msg_buf, format, ..args)
	}
	msg := strings.to_string(log_msg_buf)

	st: windows.SYSTEMTIME
	GetLocalTime(&st)
	// The stamp is written straight into the buffer: fmt goes through reflection
	// for every argument and this runs for every line, so six integers cost more
	// than the rest of the line put together. The bytes are the same.
	strings.builder_reset(&log_line_buf)
	log_write_stamp(&log_line_buf, st)
	strings.write_string(&log_line_buf, msg)
	strings.write_byte(&log_line_buf, '\n')
	line := strings.to_string(log_line_buf)
	// The terminal mirror is one write per line, not two: the newline goes into
	// the same buffer, so the round trip to the console costs one syscall.
	strings.write_byte(&log_msg_buf, '\n')
	mirror := strings.to_string(log_msg_buf)

	// A line longer than the buffer is written straight through.
	if log_backlog_len + len(line) > len(log_backlog) {
		log_flush_unlocked()
		if len(line) > len(log_backlog) {
			if log_file == nil {
				if f, err := os.open(log_path, {.Append, .Create}); err == nil { log_file = f }
			}
			if log_file != nil {
				if _, err := os.write_string(log_file, line); err != nil {
					os.close(log_file)
					log_file = nil
				}
			}
			if log_mirror { fmt.print(mirror) }
			return
		}
	}
	if log_backlog_len == 0 { log_pending_since = time.now() }
	copy(log_backlog[log_backlog_len:], line)
	log_backlog_len += len(line)
	if log_backlog_len >= len(log_backlog) ||
	   time.duration_milliseconds(time.since(log_pending_since)) >= 500 {
		log_flush_unlocked()
	}
	if log_mirror { fmt.print(mirror) }
}
