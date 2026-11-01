package phonon

// AVX2/FMA/F16C kernels for the weight formats the model actually ships.
// Every kernel is a straight-line vector loop over one weight row: the
// encoder streams ~600 M weights per utterance on 4 Zen2 cores, so the
// scalar version in linalg.odin is ~15x off the reference (ggml) and these
// replace its inner loops.
//
// Activation layout matches ggml's Q8_0 mul_mat: the activation row is
// quantised once per frame (d = amax/127) and then dotted against the int8
// weight row with vpmaddubsw, so weights and activations are both int8
// and the products land in i32 before the per-block scale.

import "base:intrinsics"
import "core:math"
import "core:simd"

// LLVM intrinsics core:simd does not expose.
@(private, default_calling_convention = "none")
foreign _ {
	@(link_name = "llvm.x86.avx2.pmadd.ub.sw")
	qk_pmaddubsw :: proc(a: simd.i8x32, b: simd.i8x32) -> simd.i16x16 ---
	@(link_name = "llvm.x86.avx2.pmadd.wd")
	qk_pmaddwd :: proc(a, b: simd.i16x16) -> simd.i32x8 ---
}

@(private, require_results)
qk_load8 :: #force_inline proc "contextless" (p: rawptr) -> simd.f32x8 {
	v := (cast(^[8]f32)p)^
	return transmute(simd.f32x8)v
}

@(private, require_results)
qk_load_i8x32 :: #force_inline proc "contextless" (p: rawptr) -> simd.i8x32 {
	v := (cast(^[32]i8)p)^
	return transmute(simd.i8x32)v
}

// w * sign(x): -w where x < 0, w elsewhere. Select over a negation is what
// LLVM turns into vpsignb, one uop instead of the (w ^ (x>>7)) - (x>>7)
// sequence it used to emit.
@(private, require_results, enable_target_feature = "avx2")
qk_signmul :: #force_inline proc "contextless" (w, x: simd.i8x32) -> simd.i8x32 {
	return simd.select(simd.lanes_lt(x, simd.i8x32(0)), simd.sub(simd.i8x32(0), w), w)
}

// Act is one quantised activation row: 32 int8 per block plus the block scale.
Act :: struct {
	q: []i8,
	d: []f32,
}

act_blocks :: proc(cols: int) -> int {
	return cols / Q8_0_QUANT_SIZEOF
}

act_make :: proc(rows, cols: int, alloc := context.allocator) -> Act {
	blocks := act_blocks(cols)
	return Act{q = make([]i8, rows * cols, alloc), d = make([]f32, rows * blocks, alloc)}
}

act_destroy :: proc(a: ^Act, alloc := context.allocator) {
	if a.q != nil { delete(a.q, alloc) }
	if a.d != nil { delete(a.d, alloc) }
	a^ = {}
}

// float_to_half rounds f32 to f16 (nearest even), matching
// GGML_FP32_TO_FP16, so stored scales reproduce ggml bit for bit.
@(private)
float_to_half :: #force_inline proc "contextless" (f: f32) -> u16 {
	u := transmute(u32)f
	sign := u16((u >> 16) & 0x8000)
	exp := i32((u >> 23) & 0xff)
	mant := u & 0x7fffff
	if exp == 0xff { // inf / nan
		return sign | 0x7c00 | u16(mant >> 13)
	}
	e := exp - 127 + 15
	if e >= 0x1f {
		return sign | 0x7c00 // overflow -> inf
	}
	if e <= 0 {
		// subnormal (or underflow to zero): shift the implicit bit back in
		if e < -10 { return sign }
		m := mant | 0x800000
		shift := u32(14 - e)
		half := m >> shift
		rem := m & ((u32(1) << shift) - 1)
		mid := u32(1) << (shift - 1)
		if rem > mid || (rem == mid && (half & 1) != 0) { half += 1 }
		return sign | u16(half)
	}
	half := u16((u32(e) << 10) | (mant >> 13))
	rem := mant & 0x1fff
	if rem > 0x1000 || (rem == 0x1000 && (half & 1) != 0) { half += 1 }
	return sign | half
}

// quantize_row_q8_0 mirrors ggml: per 32-value block d = amax/127, and the
// scale is *stored as f16* (GGML_FP32_TO_FP16) even though the int8 values
// are rounded with the exact f32 d. Keeping d in f32 here makes every dot
// product differ from ggml by up to 5e-4 relative per block, which the
// encoder's LayerNorms amplify well past the decoding margin.
@(enable_target_feature = "avx2")
quantize_row_q8_0 :: proc(x: []f32, q: []i8, d: []f32) {
	for b in 0 ..< len(x) / Q8_0_QUANT_SIZEOF {
		off := b * Q8_0_QUANT_SIZEOF
		amax := f32(0)
		for i in 0 ..< Q8_0_QUANT_SIZEOF {
			v := math.abs(x[off + i])
			if v > amax { amax = v }
		}
		dd := amax / 127.0
		id := dd != 0 ? 1.0 / dd : 0
		d[b] = half_to_float(float_to_half(dd))
		for i in 0 ..< Q8_0_QUANT_SIZEOF {
			q[off + i] = i8(math.round(x[off + i] * id))
		}
	}
}

// quantize_acts quantises every frame of x (one 32-block row each).
@(enable_target_feature = "avx2")
quantize_acts :: proc(a: Act, x: Mat) {
	cols := x.cols
	blocks := act_blocks(cols)
	for f in 0 ..< x.rows {
		quantize_row_q8_0(
			x.data[f * cols:][:cols],
			a.q[f * cols:][:cols],
			a.d[f * blocks:][:blocks],
		)
	}
}

// qk_pf returns one block's 32 MACs as 8 f32 lanes, unscaled: the caller
// folds the block scale into its accumulator with a single fma, so a block
// costs two loads, the sign multiply, vpmaddubsw+vpmaddwd, vcvtdq2ps and one
// fma. Four blocks run independently, which hides the 13-cycle madd->cvt->fma
// chain behind throughput.
@(private, require_results, enable_target_feature = "avx2")
qk_pf :: #force_inline proc "contextless" (woff, qoff: uintptr) -> simd.f32x8 {
	wv := qk_load_i8x32(rawptr(woff + 2))
	xv := qk_load_i8x32(rawptr(qoff))
	p16 := qk_pmaddubsw(simd.abs(xv), qk_signmul(wv, xv))
	return simd.f32x8(qk_pmaddwd(p16, simd.i16x16(1)))
}

@(private, require_results)
qk_load4 :: #force_inline proc "contextless" (p: rawptr) -> simd.f32x4 {
	v := (cast(^[4]f32)p)^
	return transmute(simd.f32x4)v
}

// q8_0_dot_q8 computes w . act for one Q8_0 weight row of len(act.q) values.
// Both operands are int8; vpmaddubsw takes the activation as unsigned, so the
// activation is abs()'d and the sign folded into the weight instead. ds carries
// the row's block scales decoded at load, so four of them are one vector load.
@(enable_target_feature = "avx2,fma")
q8_0_dot_q8 :: proc "contextless" (w: []u8, ws: []f32, q: []i8, d: []f32) -> f32 {
	blocks := len(q) / Q8_0_QUANT_SIZEOF
	wp := uintptr(raw_data(w))
	wsp := uintptr(raw_data(ws))
	qp := uintptr(raw_data(q))
	dp := uintptr(raw_data(d))
	acc0 := simd.f32x8(0)
	acc1 := simd.f32x8(0)
	acc2 := simd.f32x8(0)
	acc3 := simd.f32x8(0)
	b := 0
	for ; b + 4 <= blocks; b += 4 {
		w0 := wp + uintptr(b * Q8_0_BLOCK_SIZEOF)
		q0 := qp + uintptr(b * Q8_0_QUANT_SIZEOF)
		d0 := dp + uintptr(b * 4)
		dw := qk_load4(rawptr(wsp + uintptr(b * 4))) * qk_load4(rawptr(d0))
		dh := transmute([4]f32)dw
		acc0 = simd.fma(qk_pf(w0, q0), simd.f32x8(dh[0]), acc0)
		acc1 = simd.fma(qk_pf(w0 + 34, q0 + 32), simd.f32x8(dh[1]), acc1)
		acc2 = simd.fma(qk_pf(w0 + 68, q0 + 64), simd.f32x8(dh[2]), acc2)
		acc3 = simd.fma(qk_pf(w0 + 102, q0 + 96), simd.f32x8(dh[3]), acc3)
	}
	acc := (acc0 + acc1) + (acc2 + acc3)
	for ; b < blocks; b += 1 {
		dh := (cast(^f32)(wsp + uintptr(b * 4)))^ * (cast(^f32)(dp + uintptr(b * 4)))^
		acc = simd.fma(qk_pf(wp + uintptr(b * Q8_0_BLOCK_SIZEOF), qp + uintptr(b * Q8_0_QUANT_SIZEOF)), simd.f32x8(dh), acc)
	}
	return simd.reduce_add_ordered(acc)
}

// f32_dot_avx computes w . x for one F32 weight row.
@(enable_target_feature = "avx2,fma")
f32_dot_avx :: proc "contextless" (w: []u8, x: []f32) -> f32 {
	n := len(x)
	wp := uintptr(raw_data(w))
	xp := uintptr(raw_data(x))
	acc0 := simd.f32x8(0)
	acc1 := simd.f32x8(0)
	acc2 := simd.f32x8(0)
	acc3 := simd.f32x8(0)
	i := 0
	for ; i + 32 <= n; i += 32 {
		o := uintptr(i * 4)
		acc0 = simd.fma(qk_load8(rawptr(wp + o)), qk_load8(rawptr(xp + o)), acc0)
		acc1 = simd.fma(qk_load8(rawptr(wp + o + 32)), qk_load8(rawptr(xp + o + 32)), acc1)
		acc2 = simd.fma(qk_load8(rawptr(wp + o + 64)), qk_load8(rawptr(xp + o + 64)), acc2)
		acc3 = simd.fma(qk_load8(rawptr(wp + o + 96)), qk_load8(rawptr(xp + o + 96)), acc3)
	}
	for ; i + 8 <= n; i += 8 {
		acc0 = simd.fma(qk_load8(rawptr(wp + uintptr(i*4))), qk_load8(rawptr(xp + uintptr(i*4))), acc0)
	}
	sum := simd.reduce_add_ordered((acc0 + acc1) + (acc2 + acc3))
	for ; i < n; i += 1 {
		sum += (cast(^f32)(wp + uintptr(i * 4)))^ * x[i]
	}
	return sum
}