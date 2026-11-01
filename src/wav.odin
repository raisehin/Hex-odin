package main

import "core:mem"
import "core:os"

// Minimal RIFF/WAVE writer for f32 mono input; converts to 16-bit PCM so
// transcribe.cpp and every other tool accept it.
wav_write_f32 :: proc(path: string, samples: []f32, rate: u32, channels: u16) -> bool {
	data_bytes := len(samples) * 2
	hdr := make([]byte, 44, context.temp_allocator)
	put_u32 := proc(b: []byte, off: int, v: u32) {
		b[off+0] = byte(v); b[off+1] = byte(v >> 8); b[off+2] = byte(v >> 16); b[off+3] = byte(v >> 24)
	}
	put_u16 := proc(b: []byte, off: int, v: u16) {
		b[off+0] = byte(v); b[off+1] = byte(v >> 8)
	}
	copy(hdr[0:4], "RIFF")
	put_u32(hdr, 4, u32(36 + data_bytes))
	copy(hdr[8:12], "WAVE")
	copy(hdr[12:16], "fmt ")
	put_u32(hdr, 16, 16)
	put_u16(hdr, 20, 1) // PCM
	put_u16(hdr, 22, channels)
	put_u32(hdr, 24, rate)
	put_u32(hdr, 28, rate * u32(channels) * 2)
	put_u16(hdr, 32, channels * 2)
	put_u16(hdr, 34, 16)
	copy(hdr[36:40], "data")
	put_u32(hdr, 40, u32(data_bytes))

	f, err := os.create(path)
	if err != nil { return false }
	defer os.close(f)
	if _, e := os.write(f, hdr); e != nil { return false }
	pcm := make([]i16, len(samples), context.temp_allocator)
	for s, i in samples {
		v := s
		if v > 1 { v = 1 }
		if v < -1 { v = -1 }
		pcm[i] = i16(v * 32767)
	}
	bytes := mem.slice_to_bytes(pcm)
	_, e := os.write(f, bytes)
	return e == nil
}

// Reads a 16-bit PCM wav, downmixes to mono and linearly resamples to 16 kHz.
wav_read_mono16k :: proc(path: string) -> (samples: []f32, rate: u32, ok: bool) {
	data, err := os.read_entire_file(path, context.allocator)
	if err != nil || len(data) < 44 { return nil, 0, false }
	get_u16 :: proc(b: []byte, o: int) -> u16 { return u16(b[o]) | u16(b[o+1]) << 8 }
	get_u32 :: proc(b: []byte, o: int) -> u32 {
		return u32(b[o]) | u32(b[o+1]) << 8 | u32(b[o+2]) << 16 | u32(b[o+3]) << 24
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" { return nil, 0, false }

	channels: u16 = 1
	bits: u16 = 16
	src_rate: u32 = 16000
	off := 12
	frames: []i16
	for off + 8 <= len(data) {
		id := string(data[off:off+4])
		size := int(get_u32(data, off + 4))
		body := off + 8
		if body + size > len(data) { size = len(data) - body }
		if id == "fmt " {
			channels = get_u16(data, body + 2)
			src_rate = get_u32(data, body + 4)
			bits = get_u16(data, body + 14)
		} else if id == "data" {
			if bits == 16 {
				n := size / 2
				frames = make([]i16, n, context.temp_allocator)
				for i in 0 ..< n {
					frames[i] = i16(get_u16(data, body + i*2))
				}
			}
		}
		off = body + size + (size & 1)
	}
	if len(frames) == 0 || channels == 0 { return nil, 0, false }

	mono := make([]f32, len(frames) / int(channels), context.temp_allocator)
	for i in 0 ..< len(mono) {
		sum: f32 = 0
		for c in 0 ..< int(channels) { sum += f32(frames[i*int(channels) + c]) / 32768.0 }
		mono[i] = sum / f32(channels)
	}

	out_rate: u32 = 16000
	n_out := int(f64(len(mono)) * f64(out_rate) / f64(src_rate))
	out := make([]f32, n_out)
	for i in 0 ..< n_out {
		src := f32(i) * f32(src_rate) / f32(out_rate)
		i0 := int(src)
		t := src - f32(i0)
		a := mono[min(i0, len(mono)-1)]
		b := mono[min(i0+1, len(mono)-1)]
		out[i] = a + (b - a) * t
	}
	return out, out_rate, true
}
