package main

import "core:math"
import "core:strings"
import "core:sync"

import ma "vendor:miniaudio"

// Capture goes through miniaudio (vendored), which handles WASAPI setup,
// resampling to 16 kHz mono f32 and device hot-swapping. The previous direct
// IAudioCaptureClient path is gone: one callback, one ring buffer.
CAPTURE_RATE :: 16000
RING_SAMPLES :: CAPTURE_RATE * 60 * 2 // 2 minutes; dictation, not meetings

Capture_Error :: enum { None, Context, Device, Start }

Capture_Level :: struct {
	rms:    f32,
	peak:   f32,
	frames: u64,
}

Device_Info :: struct {
	name:       string,
	is_default: bool,
}

Capture :: struct {
	rate: u32,
}

capture_ready: bool
cap_context: ma.context_type
cap_device: ma.device

cap_ring: [RING_SAMPLES]f32
cap_write: i64
cap_read: i64

capture_start :: proc() -> (c: Capture, err: Capture_Error) {
	if capture_ready { return Capture{rate = CAPTURE_RATE}, .None }

	if ma.context_init(nil, 0, nil, &cap_context) != .SUCCESS { return c, .Context }

	cfg := ma.device_config_init(.capture)
	cfg.sampleRate = CAPTURE_RATE
	cfg.capture.format = .f32
	cfg.capture.channels = 1
	cfg.periodSizeInMilliseconds = 20
	cfg.dataCallback = cap_callback

	if ma.device_init(&cap_context, &cfg, &cap_device) != .SUCCESS {
		ma.context_uninit(&cap_context)
		return c, .Device
	}
	if ma.device_start(&cap_device) != .SUCCESS {
		ma.device_uninit(&cap_device)
		ma.context_uninit(&cap_context)
		return c, .Start
	}

	sync.atomic_store(&cap_write, 0)
	sync.atomic_store(&cap_read, 0)
	capture_ready = true
	return Capture{rate = CAPTURE_RATE}, .None
}

// Drops everything captured so far: the pump only runs while recording, so the
// ring holds stale audio at the moment the hotkey is pressed.
capture_discard :: proc() {
	w := sync.atomic_load(&cap_write)
	sync.atomic_store(&cap_read, w)
}

capture_stop :: proc(c: ^Capture) {
	_ = c
	if !capture_ready { return }
	capture_ready = false
	ma.device_uninit(&cap_device)
	ma.context_uninit(&cap_context)
}

cap_callback :: proc "c" (device: ^ma.device, output, input: rawptr, frameCount: u32) {
	if input == nil { return }
	samples := cast([^]f32)input
	w := sync.atomic_load(&cap_write)
	start := w
	for i in 0 ..< int(frameCount) {
		idx := (start + i64(i)) % RING_SAMPLES
		cap_ring[idx] = samples[i]
		// Castagnoli-ish cheap limiter: keep the ring bounded if nobody drains.
	}
	sync.atomic_store(&cap_write, w + i64(frameCount))
	_ = device
	_ = output
}

// Moves everything the callback wrote into out, updating level.
capture_pump :: proc(c: ^Capture, out: ^[dynamic]f32, level: ^Capture_Level) -> bool {
	_ = c
	w := sync.atomic_load(&cap_write)
	r := sync.atomic_load(&cap_read)
	if w - r > RING_SAMPLES {
		// Reader fell behind: drop the oldest so the writer never stalls.
		r = w - RING_SAMPLES
		sync.atomic_store(&cap_read, r)
	}
	if w == r { return false }
	drained := w - r
	// Grow once for the whole period instead of reallocating per sample.
	reserve(out, len(out^) + int(drained))
	// The ring is circular, so the run is at most two memcpys. Scanning the
	// appended slice afterwards keeps the RMS/peak math contiguous and drops the
	// per-sample modulo division.
	old_len := len(out^)
	start := int(r % RING_SAMPLES)
	first := RING_SAMPLES - start
	if int(drained) < first { first = int(drained) }
	rest := int(drained) - first
	resize(out, old_len + int(drained))
	copy(out^[old_len:old_len+first], cap_ring[start:start+first])
	if rest > 0 {
		copy(out^[old_len+first:], cap_ring[0:rest])
	}
	sum: f64
	peak: f32
	for v in out^[old_len:] {
		sum += f64(v) * f64(v)
		if a := abs(v); a > peak { peak = a }
	}
	sync.atomic_store(&cap_read, w)
	if level == nil { return true }
	// level.frames is the count drained this pump, not len(out), which keeps the
	// whole recording and over-counts by the buffer length on every pump.
	level.frames += u64(drained)
	if drained > 0 {
		level.rms = f32(math.sqrt(sum / f64(drained)))
	}
	level.peak = peak
	return true
}

capture_devices :: proc() -> []Device_Info {
	out: [dynamic]Device_Info
	ctx: ma.context_type
	if ma.context_init(nil, 0, nil, &ctx) != .SUCCESS { return out[:] }
	defer ma.context_uninit(&ctx)

	captures: [^]ma.device_info
	count: u32
	if ma.context_get_devices(&ctx, nil, nil, &captures, &count) != .SUCCESS { return out[:] }
	for i in 0 ..< int(count) {
		d := captures[i]
		n := 0
		for n < len(d.name) - 1 && d.name[n] != 0 { n += 1 }
		append(&out, Device_Info{name = strings.clone(string(d.name[:n])), is_default = bool(d.isDefault)})
	}
	return out[:]
}
