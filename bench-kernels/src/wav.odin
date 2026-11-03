package main

import "core:mem"
import "core:os"

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
