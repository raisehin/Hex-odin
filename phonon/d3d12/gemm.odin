// Q8_0 GEMM on D3D12, matching phonon's CPU weight/activation layout.
//
// Two lifetimes because the encoder reuses weights across utterances but
// resizes activations per utterance:
//
//   Weights - one tensor's repacked Q8_0 rows, uploaded once and read by
//             every utterance until the engine closes. The GGUF payload
//             interleaves a f16 scale with each 32-byte block, which makes
//             every other block start at address 2 mod 4; the shader wants
//             aligned 4-byte loads and a warp-contiguous row, so
//             weights_upload splits each row into quants + f32 scales.
//   Scratch - per-(n, k, t) activation, output and readback buffers.
//
// Gemm is the single-shape convenience wrapper the microbench uses.

package phonon_d3d12

import "core:fmt"
import "core:mem"
import "vendor:directx/d3d12"

// Compiled from q8_0_gemm.hlsl by build_shaders.ps1.
Q8_0_GEMM_DXIL: []u8 = #load("q8_0_gemm.dxil")

MAX_K :: 4096
WARPS_PER_GROUP :: 4
// Must mirror FRAMES_PER_WARP in q8_0_gemm.hlsl: it sizes groups[1] and the
// per-warp frame tile. Changing one without the other silently mislaunches.
FRAMES_PER_WARP :: 8

Weights :: struct {
	g:         ^Gpu,
	n:         int, // output rows
	k:         int, // input cols
	blocks:    int,
	row_bytes: int,
	has_bias:  bool,
	q:         ^Buffer, // [n][k] int8
	d:         ^Buffer, // [n][blocks] f32
	bias:      ^Buffer, // [n] f32, or 4 dummy bytes when has_bias is false
}

weights_upload :: proc(g: ^Gpu, n, k: int, bytes: []u8, has_bias: bool) -> (w: ^Weights, err: string) {
	if n <= 0 || k <= 0 { return nil, "weights_upload: empty shape" }
	if k % 32 != 0 { return nil, fmt.tprintf("weights_upload: k=%d is not a multiple of 32", k) }
	if k > MAX_K { return nil, fmt.tprintf("weights_upload: k=%d exceeds shader MAX_K=%d", k, MAX_K) }

	w = new(Weights)
	w.g = g
	w.n, w.k = n, k
	w.blocks = k / 32
	w.row_bytes = w.blocks * 34
	w.has_bias = has_bias

	want := n * w.row_bytes
	if len(bytes) != want {
		free(w)
		return nil, fmt.tprintf("weights_upload: %d bytes, want %d", len(bytes), want)
	}
	if w.q, err = buffer_create(g, u64(n*k), .UPLOAD, d3d12.RESOURCE_STATE_GENERIC_READ); err != "" {
		free(w); return nil, err
	}
	if w.d, err = buffer_create(g, u64(n*w.blocks*4), .UPLOAD, d3d12.RESOURCE_STATE_GENERIC_READ); err != "" {
		weights_destroy(w); return nil, err
	}
	bias_bytes := u64(4)
	if has_bias { bias_bytes = u64(n * 4) }
	if w.bias, err = buffer_create(g, bias_bytes, .UPLOAD, d3d12.RESOURCE_STATE_GENERIC_READ); err != "" {
		weights_destroy(w); return nil, err
	}

	q := cast([^]u8)w.q.mapped
	d := cast([^]f32)w.d.mapped
	for o in 0 ..< n {
		src_row := o * w.row_bytes
		q_row := o * k
		d_row := o * w.blocks
		for b in 0 ..< w.blocks {
			src := src_row + b * 34
			bits := u16(bytes[src]) | u16(bytes[src + 1]) << 8
			d[d_row + b] = f16_to_f32(bits)
			mem.copy(&q[q_row + b * 32], &bytes[src + 2], 32)
		}
	}
	return w, ""
}

weights_set_bias :: proc(w: ^Weights, bias: []f32) -> string {
	if !w.has_bias { return "" }
	if len(bias) != w.n { return fmt.tprintf("weights_set_bias: %d values, want %d", len(bias), w.n) }
	return buffer_write(w.bias, mem.slice_data_cast([]u8, bias))
}

weights_destroy :: proc(w: ^Weights) {
	if w == nil { return }
	buffer_destroy(w.q)
	buffer_destroy(w.d)
	buffer_destroy(w.bias)
	free(w)
}

Scratch :: struct {
	g:         ^Gpu,
	n, k, t:  int,
	blocks:    int,
	act_q:     ^Buffer, // [t][k] int8
	act_d:     ^Buffer, // [t][blocks] f32
	out:       ^Buffer, // [t][n] f32 device-side (slots array for batched use)
	readback:  ^Buffer, // [t][n] f32 host-visible (slots array for batched use)
	out_state: d3d12.RESOURCE_STATES,
	slots:     int, // how many [t][n] output slots out/readback hold
}

scratch_create :: proc(g: ^Gpu, n, k, t: int) -> (s: ^Scratch, err: string) {
	if n <= 0 || k <= 0 || t <= 0 { return nil, "scratch_create: empty shape" }
	s = new(Scratch)
	s.g = g
	s.n, s.k, s.t = n, k, t
	s.blocks = k / 32
	if s.act_q, err = buffer_create(g, u64(t*k), .UPLOAD, d3d12.RESOURCE_STATE_GENERIC_READ); err != "" {
		free(s); return nil, err
	}
	if s.act_d, err = buffer_create(g, u64(t*s.blocks*4), .UPLOAD, d3d12.RESOURCE_STATE_GENERIC_READ); err != "" {
		scratch_destroy(s); return nil, err
	}
	if s.out, err = buffer_create(g, u64(t*n*4), .DEFAULT, STATE_COMMON, {.ALLOW_UNORDERED_ACCESS}); err != "" {
		scratch_destroy(s); return nil, err
	}
	if s.readback, err = buffer_create(g, u64(t*n*4), .READBACK, STATE_COPY_DEST); err != "" {
		scratch_destroy(s); return nil, err
	}
	s.out_state = STATE_COMMON
	s.slots = 1
	return s, ""
}

scratch_destroy :: proc(s: ^Scratch) {
	if s == nil { return }
	buffer_destroy(s.act_q)
	buffer_destroy(s.act_d)
	buffer_destroy(s.out)
	buffer_destroy(s.readback)
	free(s)
}

// gemm_exec runs one GEMM and returns a freshly allocated [t][n] result.
gemm_exec :: proc(g: ^Gpu, w: ^Weights, s: ^Scratch) -> (out: []f32, err: string) {
	out = make([]f32, s.t * s.n)
	if err = gemm_exec_into(g, w, s, out); err != "" {
		delete(out)
		return nil, err
	}
	return out, ""
}

// gemm_constants builds the b0 payload and group counts for (w, s, n_out).
@(private = "file")
gemm_constants :: proc(w: ^Weights, s: ^Scratch, n_out: int) -> ([ROOT_CONSTANT_COUNT]u32, [3]u32) {
	constants := [ROOT_CONSTANT_COUNT]u32 {
		u32(n_out), u32(s.k), u32(s.t),
		w.has_bias ? 1 : 0,
		u32(s.blocks), 0, 0, 0,
	}
	groups := [3]u32 {
		u32((n_out + WARPS_PER_GROUP - 1) / WARPS_PER_GROUP),
		u32((s.t + FRAMES_PER_WARP - 1) / FRAMES_PER_WARP),
		1,
	}
	return constants, groups
}

// scratch_ensure_slots grows the scratch to hold nslots batched outputs.
scratch_ensure_slots :: proc(s: ^Scratch, nslots: int) -> string {
	if s.slots >= nslots { return "" }
	g := s.g
	if s.out != nil { buffer_destroy(s.out) }
	if s.readback != nil { buffer_destroy(s.readback) }
	s.out = nil
	s.readback = nil
	bytes := u64(nslots * s.t * s.n * 4)
	out, err := buffer_create(g, bytes, .DEFAULT, STATE_COMMON, {.ALLOW_UNORDERED_ACCESS})
	if err != "" { return err }
	rb, rerr := buffer_create(g, bytes, .READBACK, STATE_COPY_DEST)
	if rerr != "" { buffer_destroy(out); return rerr }
	s.out = out
	s.readback = rb
	s.out_state = STATE_COMMON
	s.slots = nslots
	return ""
}

// gemm_launch_batch3 submits three GEMMs that share one activation buffer
// (the q/k/v projections) into a single command list with one fence signal.
// Each lands in its own output slot, so the three CPU halves can run
// back-to-back and one join covers all three.
// Numerically identical: same shader, same constants, same row range.
gemm_launch_batch3 :: proc(g: ^Gpu, ws: [3]^Weights, s: ^Scratch, n_out: int) -> string {
	if n_out <= 0 { return fmt.tprintf("gemm_launch_batch3: n_out=%d", n_out) }
	if s.slots < 3 { return fmt.tprintf("gemm_launch_batch3: scratch has %d slots, want 3", s.slots) }
	entries: [3]Dispatch_Entry
	for w, i in ws {
		if w == nil { return "gemm_launch_batch3: nil weights" }
		if n_out > w.n { return fmt.tprintf("gemm_launch_batch3: n_out=%d for %d rows", n_out, w.n) }
		if s.n != n_out || s.k != w.k { return "gemm_launch_batch3: weights/scratch shape mismatch" }
		constants, groups := gemm_constants(w, s, n_out)
		entries[i] = Dispatch_Entry {
			groups    = groups,
			constants = constants,
			srvs      = [ROOT_SRV_COUNT]^Buffer{w.q, w.d, s.act_q, s.act_d, w.bias},
		}
	}
	slot_bytes := u64(s.t * n_out * 4)
	err := dispatch_launch_batch(g, entries[:], s.out, s.out_state, s.readback, slot_bytes)
	if err == "" { s.out_state = STATE_COPY_SOURCE }
	return err
}

// gemm_finish_strided_slot is gemm_finish_strided for one slot of a batched
// submission: waits once and scatters that slot's rows into dst pitched by
// n_full.
gemm_finish_strided_slot :: proc(g: ^Gpu, s: ^Scratch, dst: []f32, n_full, slot: int) -> (err: string) {
	t, mid := s.t, s.n
	if len(dst) != t * n_full { return fmt.tprintf("gemm_finish_strided_slot: dst has %d, want %d", len(dst), t * n_full) }
	if mid > n_full { return fmt.tprintf("gemm_finish_strided_slot: mid %d exceeds pitch %d", mid, n_full) }
	if s.readback.mapped == nil { return "gemm_finish_strided_slot: readback not mapped" }
	if err = gpu_join(g); err != "" { return err }
	slot_bytes := u64(t * mid * 4)
	base := cast([^]u8)s.readback.mapped
	src := base[uintptr(u64(slot)*slot_bytes):]
	for f in 0 ..< t {
		mem.copy(&dst[f*n_full], rawptr(&src[uintptr(f*mid*4)]), mid*4)
	}
	return ""
}

// gemm_launch submits one GEMM for output rows [0, n_out) without waiting,
// so the caller can run the CPU half of a row-split concurrently. Pair with
// gemm_finish. n_out must not exceed the uploaded weight rows.
gemm_launch :: proc(g: ^Gpu, w: ^Weights, s: ^Scratch, n_out: int) -> (err: string) {
	if n_out <= 0 || n_out > w.n { return fmt.tprintf("gemm_launch: n_out=%d for %d weight rows", n_out, w.n) }
	if s.n != n_out || s.k != w.k { return "gemm_launch: weights/scratch shape mismatch" }
	constants, groups := gemm_constants(w, s, n_out)
	srvs := [ROOT_SRV_COUNT]^Buffer{w.q, w.d, s.act_q, s.act_d, w.bias}
	out_bytes := s.t * n_out * 4
	err = dispatch_launch(g, groups, &constants, srvs, s.out, s.out_state, s.out, s.readback, out_bytes)
	if err == "" { s.out_state = STATE_COPY_SOURCE }
	return err
}

// gemm_finish waits for a launched GEMM and reads [t][n] floats into dst.
gemm_finish :: proc(g: ^Gpu, s: ^Scratch, dst: []f32) -> (err: string) {
	if len(dst) != s.t * s.n { return fmt.tprintf("gemm_finish: dst has %d, want %d", len(dst), s.t * s.n) }
	return dispatch_finish(g, s.readback, mem.slice_data_cast([]u8, dst))
}

// scratch_write_acts uploads one split linear's quantized activations with
// a single validation instead of two buffer_write calls. The bytes landing
// in act_q/act_d are identical; only the per-call overhead is removed.
scratch_write_acts :: proc(s: ^Scratch, q: []u8, d: []u8) -> string {
	if s.act_q.mapped == nil || s.act_d.mapped == nil { return "scratch_write_acts on a non-mapped buffer" }
	if u64(len(q)) > s.act_q.size { return fmt.tprintf("scratch_write_acts q overflow: %d > %d", len(q), s.act_q.size) }
	if u64(len(d)) > s.act_d.size { return fmt.tprintf("scratch_write_acts d overflow: %d > %d", len(d), s.act_d.size) }
	mem.copy(s.act_q.mapped, raw_data(q), len(q))
	mem.copy(s.act_d.mapped, raw_data(d), len(d))
	return ""
}

// gemm_finish_strided waits for a launched GEMM and scatters its [t][s.n]
// rows straight into dst pitched by n_full, replacing the tmp buffer +
// re-copy pass with a single pass over the same bytes. Bit-identical: the
// floats are only written to a new address, never recomputed.
gemm_finish_strided :: proc(g: ^Gpu, s: ^Scratch, dst: []f32, n_full: int) -> (err: string) {
	t, mid := s.t, s.n
	if len(dst) != t * n_full { return fmt.tprintf("gemm_finish_strided: dst has %d, want %d", len(dst), t * n_full) }
	if mid > n_full { return fmt.tprintf("gemm_finish_strided: mid %d exceeds pitch %d", mid, n_full) }
	if s.readback.mapped == nil { return "gemm_finish_strided: readback not mapped" }
	if err = gpu_join(g); err != "" { return err }
	base := cast([^]u8)s.readback.mapped
	for f in 0 ..< t {
		mem.copy(&dst[f*n_full], &base[f*mid*4], mid*4)
	}
	return ""
}

// gemm_exec_into runs one GEMM into the caller's [t][n] buffer, so the encoder
// hot loop pays no allocator traffic. The GPU wait and the readback are part
// of the call: the encoder consumes the result on the CPU immediately.
gemm_exec_into :: proc(g: ^Gpu, w: ^Weights, s: ^Scratch, dst: []f32) -> (err: string) {
	if w.n != s.n || w.k != s.k { return "gemm_exec_into: weights/scratch shape mismatch" }
	if err = gemm_launch(g, w, s, s.n); err != "" { return err }
	return gemm_finish(g, s, dst)
}

@(private = "file")
f16_to_f32 :: proc(h: u16) -> f32 {
	sign := u32(h & 0x8000) << 16
	exp := u32(h >> 10) & 0x1F
	mant := u32(h & 0x03FF)
	if exp == 0 {
		if mant == 0 { return transmute(f32)sign }
		// Subnormal: renormalise into the f32 exponent range.
		for (mant & 0x0400) == 0 {
			mant <<= 1
			exp -= 1
		}
		exp += 1
		mant &= 0x03FF
		return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
	}
	if exp == 31 { return transmute(f32)(sign | 0x7F80_0000 | (mant << 13)) }
	return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
}

// Gemm is the single-shape wrapper the microbench uses: one tensor, one
// frame count, weights uploaded once, activations re-uploaded per run.
Gemm :: struct {
	g:        ^Gpu,
	n, k, t: int,
	has_bias: bool,
	w:        ^Weights,
	s:        ^Scratch,
}

gemm_create :: proc(g: ^Gpu, n, k, t: int, has_bias: bool) -> (m: ^Gemm, err: string) {
	m = new(Gemm)
	m.g = g
	m.n, m.k, m.t = n, k, t
	m.has_bias = has_bias
	if m.s, err = scratch_create(g, n, k, t); err != "" {
		free(m)
		return nil, err
	}
	return m, ""
}

gemm_set_weights :: proc(m: ^Gemm, bytes: []u8) -> string {
	if m.w != nil { weights_destroy(m.w); m.w = nil }
	w, err := weights_upload(m.g, m.n, m.k, bytes, m.has_bias)
	if err != "" { return err }
	m.w = w
	return ""
}

gemm_set_activations :: proc(m: ^Gemm, q: []i8, d: []f32) -> string {
	if m.s == nil { return "gemm_set_activations: no scratch" }
	if len(q) != m.t*m.k { return fmt.tprintf("gemm_set_activations: q has %d, want %d", len(q), m.t*m.k) }
	if len(d) != m.t*m.s.blocks { return fmt.tprintf("gemm_set_activations: d has %d, want %d", len(d), m.t*m.s.blocks) }
	if err := buffer_write(m.s.act_q, mem.slice_data_cast([]u8, q)); err != "" { return err }
	return buffer_write(m.s.act_d, mem.slice_data_cast([]u8, d))
}

gemm_set_bias :: proc(m: ^Gemm, bias: []f32) -> string {
	if m.w == nil { return "gemm_set_bias: no weights" }
	return weights_set_bias(m.w, bias)
}

gemm_run :: proc(m: ^Gemm) -> (out: []f32, err: string) {
	if m.w == nil { return nil, "gemm_run: no weights" }
	return gemm_exec(m.g, m.w, m.s)
}

gemm_destroy :: proc(m: ^Gemm) {
	if m == nil { return }
	weights_destroy(m.w)
	scratch_destroy(m.s)
	free(m)
}
