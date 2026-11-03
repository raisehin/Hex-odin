package phonon

// AVX2/FMA kernels for the weight formats the model actually ships.
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
	@(link_name = "llvm.x86.avx2.psign.b")
	qk_psignb :: proc(a, b: simd.i8x32) -> simd.i8x32 ---
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

// w * sign(x): -w where x < 0, w elsewhere. Spelled as an intrinsic because the
// select over a negation does *not* fold to vpsignb: the AVX2 loop was emitting
// vpcmpgtb + vpxor + vpsubb, three uops on the critical path of every 32-value
// block, where vpsignb is one.
@(private, require_results, enable_target_feature = "avx2")
qk_signmul :: #force_inline proc "contextless" (w, x: simd.i8x32) -> simd.i8x32 {
	return qk_psignb(w, x)
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
	return qk_pf_w(qk_load_i8x32(rawptr(woff + 2)), qoff)
}

// qk_pf_w is qk_pf for an already-loaded weight block, so a caller that dots
// one block against several activation rows loads those 32 bytes once.
@(private, require_results, enable_target_feature = "avx2")
qk_pf_w :: #force_inline proc "contextless" (wv: simd.i8x32, qoff: uintptr) -> simd.f32x8 {
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
		d0 := uintptr(b * 4)
		// One scalar product per block, broadcast straight into the fma: the
		// f32x4 form forced a vector store per group plus a store-forwarding
		// stall on the dh[n] reload.
		s0 := (cast(^f32)(wsp + d0))^ * (cast(^f32)(dp + d0))^
		s1 := (cast(^f32)(wsp + d0 + 4))^ * (cast(^f32)(dp + d0 + 4))^
		s2 := (cast(^f32)(wsp + d0 + 8))^ * (cast(^f32)(dp + d0 + 8))^
		s3 := (cast(^f32)(wsp + d0 + 12))^ * (cast(^f32)(dp + d0 + 12))^
		acc0 = simd.fma(qk_pf(w0, q0), simd.f32x8(s0), acc0)
		acc1 = simd.fma(qk_pf(w0 + 34, q0 + 32), simd.f32x8(s1), acc1)
		acc2 = simd.fma(qk_pf(w0 + 68, q0 + 64), simd.f32x8(s2), acc2)
		acc3 = simd.fma(qk_pf(w0 + 102, q0 + 96), simd.f32x8(s3), acc3)
	}
	acc := (acc0 + acc1) + (acc2 + acc3)
	for ; b < blocks; b += 1 {
		dh := (cast(^f32)(wsp + uintptr(b * 4)))^ * (cast(^f32)(dp + uintptr(b * 4)))^
		acc = simd.fma(qk_pf(wp + uintptr(b * Q8_0_BLOCK_SIZEOF), qp + uintptr(b * Q8_0_QUANT_SIZEOF)), simd.f32x8(dh), acc)
	}
	return simd.reduce_add_ordered(acc)
}

// q8_0_dot_q8_2f computes w . act for one Q8_0 weight row against two
// quantised frames at once. The row's 34-byte blocks are the traffic that
// matters: lin_task visits every output row once per frame, so the encoder
// re-reads its ~600 MB of Q8_0 weights 11 times per utterance and the matmul
// path saturates DRAM (11.7 GFLOP at 63 GFLOPS = 35 GB/s of row bytes). One
// pass over the blocks serves both dots, halving that.
//
// Per frame the accumulation is byte-identical to q8_0_dot_q8: the same four
// interleaved accumulators over the same block quad, reduced in the same tree,
// so this cannot move the encoder output.
@(enable_target_feature = "avx2,fma")
q8_0_dot_q8_2f :: proc "contextless" (w: []u8, ws: []f32, qa: []i8, da: []f32, qb: []i8, db: []f32) -> (fa, fb: f32) {
	blocks := len(qa) / Q8_0_QUANT_SIZEOF
	wp := uintptr(raw_data(w))
	wsp := uintptr(raw_data(ws))
	qap := uintptr(raw_data(qa))
	qbp := uintptr(raw_data(qb))
	dap := uintptr(raw_data(da))
	dbp := uintptr(raw_data(db))
	a0 := simd.f32x8(0)
	a1 := simd.f32x8(0)
	a2 := simd.f32x8(0)
	a3 := simd.f32x8(0)
	b0 := simd.f32x8(0)
	b1 := simd.f32x8(0)
	b2 := simd.f32x8(0)
	b3 := simd.f32x8(0)
	b := 0
	for ; b + 4 <= blocks; b += 4 {
		w0 := wp + uintptr(b * Q8_0_BLOCK_SIZEOF)
		q0 := uintptr(b * Q8_0_QUANT_SIZEOF)
		d0 := uintptr(b * 4)
		s0 := (cast(^f32)(wsp + d0))^
		sa0 := s0 * (cast(^f32)(dap + d0))^
		sb0 := s0 * (cast(^f32)(dbp + d0))^
		s1 := (cast(^f32)(wsp + d0 + 4))^
		sa1 := s1 * (cast(^f32)(dap + d0 + 4))^
		sb1 := s1 * (cast(^f32)(dbp + d0 + 4))^
		s2 := (cast(^f32)(wsp + d0 + 8))^
		sa2 := s2 * (cast(^f32)(dap + d0 + 8))^
		sb2 := s2 * (cast(^f32)(dbp + d0 + 8))^
		s3 := (cast(^f32)(wsp + d0 + 12))^
		sa3 := s3 * (cast(^f32)(dap + d0 + 12))^
		sb3 := s3 * (cast(^f32)(dbp + d0 + 12))^
		wv := qk_load_i8x32(rawptr(w0 + 2))
		a0 = simd.fma(qk_pf_w(wv, qap + q0), simd.f32x8(sa0), a0)
		b0 = simd.fma(qk_pf_w(wv, qbp + q0), simd.f32x8(sb0), b0)
		wv = qk_load_i8x32(rawptr(w0 + 34 + 2))
		a1 = simd.fma(qk_pf_w(wv, qap + q0 + 32), simd.f32x8(sa1), a1)
		b1 = simd.fma(qk_pf_w(wv, qbp + q0 + 32), simd.f32x8(sb1), b1)
		wv = qk_load_i8x32(rawptr(w0 + 68 + 2))
		a2 = simd.fma(qk_pf_w(wv, qap + q0 + 64), simd.f32x8(sa2), a2)
		b2 = simd.fma(qk_pf_w(wv, qbp + q0 + 64), simd.f32x8(sb2), b2)
		wv = qk_load_i8x32(rawptr(w0 + 102 + 2))
		a3 = simd.fma(qk_pf_w(wv, qap + q0 + 96), simd.f32x8(sa3), a3)
		b3 = simd.fma(qk_pf_w(wv, qbp + q0 + 96), simd.f32x8(sb3), b3)
	}
	for ; b < blocks; b += 1 {
		woff := wp + uintptr(b * Q8_0_BLOCK_SIZEOF)
		qoff := uintptr(b * Q8_0_QUANT_SIZEOF)
		doff := uintptr(b * 4)
		sw := (cast(^f32)(wsp + doff))^
		wv := qk_load_i8x32(rawptr(woff + 2))
		a0 = simd.fma(qk_pf_w(wv, qap + qoff), simd.f32x8(sw * (cast(^f32)(dap + doff))^), a0)
		b0 = simd.fma(qk_pf_w(wv, qbp + qoff), simd.f32x8(sw * (cast(^f32)(dbp + doff))^), b0)
	}
	fa = simd.reduce_add_ordered((a0 + a1) + (a2 + a3))
	fb = simd.reduce_add_ordered((b0 + b1) + (b2 + b3))
	return
}

// qk_pf_w2 is qk_pf_w with the activation's sign already split out: xabs is
// abs(x) and xv the raw activation. Two weight rows dotted against the same
// frame share one xabs, one load and one abs.
@(private, require_results, enable_target_feature = "avx2")
qk_pf_w2 :: #force_inline proc "contextless" (wv, xv, xabs: simd.i8x32) -> simd.f32x8 {
	return simd.f32x8(qk_pmaddwd(qk_pmaddubsw(xabs, qk_signmul(wv, xv)), simd.i16x16(1)))
}

// q8_0_dot_q8_2f_2r dots two adjacent weight rows against two adjacent
// quantised frames: four dots per pass over the block bytes. The activation
// block of a frame is loaded and abs'd once and then reused by both rows, which
// takes the cost of a dot from ~10 vector uops to ~6.5.
//
// Each dot sums into two accumulators, one per block of a two-block step, and
// the pair is reduced in that order, so the result is a fixed function of the
// row and activation bytes. Both the serial and the pooled path call this
// kernel through lin_rows_core, which keeps their outputs bit-identical.
@(enable_target_feature = "avx2,fma")
q8_0_dot_q8_2f_2r :: proc "contextless" (
	wa: []u8, wsa: []f32,
	wb: []u8, wsb: []f32,
	q0: []i8, d0: []f32,
	q1: []i8, d1: []f32,
) -> (a0, b0, a1, b1: f32) {
	blocks := len(q0) / Q8_0_QUANT_SIZEOF
	wap := uintptr(raw_data(wa))
	wbp := uintptr(raw_data(wb))
	wsap := uintptr(raw_data(wsa))
	wsbp := uintptr(raw_data(wsb))
	q0p := uintptr(raw_data(q0))
	q1p := uintptr(raw_data(q1))
	d0p := uintptr(raw_data(d0))
	d1p := uintptr(raw_data(d1))

	aa0 := simd.f32x8(0) // row a, frame 0
	aa1 := simd.f32x8(0)
	ba0 := simd.f32x8(0) // row b, frame 0
	ba1 := simd.f32x8(0)
	ab0 := simd.f32x8(0) // row a, frame 1
	ab1 := simd.f32x8(0)
	bb0 := simd.f32x8(0) // row b, frame 1
	bb1 := simd.f32x8(0)

	b := 0
	for ; b + 2 <= blocks; b += 2 {
		wo := uintptr(b * Q8_0_BLOCK_SIZEOF)
		qo := uintptr(b * Q8_0_QUANT_SIZEOF)
		dof := uintptr(b * 4)
		{
			wva := qk_load_i8x32(rawptr(wap + wo + 2))
			wvb := qk_load_i8x32(rawptr(wbp + wo + 2))
			x0 := qk_load_i8x32(rawptr(q0p + qo))
			x1 := qk_load_i8x32(rawptr(q1p + qo))
			ax0 := simd.abs(x0)
			ax1 := simd.abs(x1)
			sa := (cast(^f32)(wsap + dof))^
			sb := (cast(^f32)(wsbp + dof))^
			da0 := (cast(^f32)(d0p + dof))^
			db0 := (cast(^f32)(d1p + dof))^
			aa0 = simd.fma(qk_pf_w2(wva, x0, ax0), simd.f32x8(sa * da0), aa0)
			ba0 = simd.fma(qk_pf_w2(wvb, x0, ax0), simd.f32x8(sb * da0), ba0)
			ab0 = simd.fma(qk_pf_w2(wva, x1, ax1), simd.f32x8(sa * db0), ab0)
			bb0 = simd.fma(qk_pf_w2(wvb, x1, ax1), simd.f32x8(sb * db0), bb0)
		}
		{
			wo := uintptr((b + 1) * Q8_0_BLOCK_SIZEOF)
			qo := uintptr((b + 1) * Q8_0_QUANT_SIZEOF)
			dof := uintptr((b + 1) * 4)
			wva := qk_load_i8x32(rawptr(wap + wo + 2))
			wvb := qk_load_i8x32(rawptr(wbp + wo + 2))
			x0 := qk_load_i8x32(rawptr(q0p + qo))
			x1 := qk_load_i8x32(rawptr(q1p + qo))
			ax0 := simd.abs(x0)
			ax1 := simd.abs(x1)
			sa := (cast(^f32)(wsap + dof))^
			sb := (cast(^f32)(wsbp + dof))^
			da0 := (cast(^f32)(d0p + dof))^
			db0 := (cast(^f32)(d1p + dof))^
			aa1 = simd.fma(qk_pf_w2(wva, x0, ax0), simd.f32x8(sa * da0), aa1)
			ba1 = simd.fma(qk_pf_w2(wvb, x0, ax0), simd.f32x8(sb * da0), ba1)
			ab1 = simd.fma(qk_pf_w2(wva, x1, ax1), simd.f32x8(sa * db0), ab1)
			bb1 = simd.fma(qk_pf_w2(wvb, x1, ax1), simd.f32x8(sb * db0), bb1)
		}
	}
	for ; b < blocks; b += 1 {
		wo := uintptr(b * Q8_0_BLOCK_SIZEOF)
		qo := uintptr(b * Q8_0_QUANT_SIZEOF)
		dof := uintptr(b * 4)
		wva := qk_load_i8x32(rawptr(wap + wo + 2))
		wvb := qk_load_i8x32(rawptr(wbp + wo + 2))
		x0 := qk_load_i8x32(rawptr(q0p + qo))
		x1 := qk_load_i8x32(rawptr(q1p + qo))
		ax0 := simd.abs(x0)
		ax1 := simd.abs(x1)
		sa := (cast(^f32)(wsap + dof))^
		sb := (cast(^f32)(wsbp + dof))^
		da0 := (cast(^f32)(d0p + dof))^
		db0 := (cast(^f32)(d1p + dof))^
		aa0 = simd.fma(qk_pf_w2(wva, x0, ax0), simd.f32x8(sa * da0), aa0)
		ba0 = simd.fma(qk_pf_w2(wvb, x0, ax0), simd.f32x8(sb * da0), ba0)
		ab0 = simd.fma(qk_pf_w2(wva, x1, ax1), simd.f32x8(sa * db0), ab0)
		bb0 = simd.fma(qk_pf_w2(wvb, x1, ax1), simd.f32x8(sb * db0), bb0)
	}
	a0 = simd.reduce_add_ordered(aa0 + aa1)
	b0 = simd.reduce_add_ordered(ba0 + ba1)
	a1 = simd.reduce_add_ordered(ab0 + ab1)
	b1 = simd.reduce_add_ordered(bb0 + bb1)
	return
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

// --- pre-encode subsampling convolutions ------------------------------------
//
// The three conv shapes in front of the encoder are small matmuls, but they
// were the slowest part of the stage: the generic path im2col'd into a
// [pos][taps] matrix, ran the frame-oriented F32 dot (9-column rows never reach
// the vector body, so 835k scalar dots for conv.0) and scattered the result back
// to [oc][pos]. Each shape now has a kernel built around the layout the next
// layer reads, so stores stay contiguous and the inner loops vectorise.

// qk_store8 stores through a [8]f32 pointer: simd.f32x8 carries 32-byte
// alignment, and the destination planes only guarantee 4.
qk_store8 :: #force_inline proc "contextless" (p: rawptr, v: simd.f32x8) {
	(cast(^[8]f32)p)^ = transmute([8]f32)v
}

Conv_First_Ctx :: struct {
	x:    []f32, // [it][ifr], one channel
	w:    []f32, // [oc][taps], tap order kx then ky
	bias: []f32,
	out:  []f32, // [oc][pos]
	pos:  int,
	it:   int,
	ot:   int,
	ofr:  int,
	ifr:  int,
	kx:   int,
	ky:   int,
	s:    int,
	p:    int,
	ocs:  int,
}

// conv_first_chunk handles one contiguous range of output positions. The taps of
// a position tile are gathered once, tap-major, so the loop over output channels
// is `taps` contiguous fma over eight positions and nothing is re-gathered.
@(enable_target_feature = "avx2,fma")
conv_first_chunk :: proc(data: rawptr, lo, hi: int) #no_bounds_check {
	c := cast(^Conv_First_Ctx)data
	TILE :: 128
	// Stack scratch: the worker threads have no allocator of their own, and the
	// taps of one tile are tiny (9 f32 for the model's 3x3 kernel).
	xm: [TILE * 16]f32
	taps := c.kx * c.ky
	xmp := uintptr(raw_data(xm[:]))
	outp := uintptr(raw_data(c.out))

	for t0 := lo; t0 < hi; t0 += TILE {
		n := min(TILE, hi - t0)
		for t in 0 ..< taps {
			ky_i := t / c.kx
			kx_i := t % c.kx
			for j in 0 ..< n {
				p := t0 + j
				iy := (p / c.ofr) * c.s - c.p + ky_i
				ix := (p % c.ofr) * c.s - c.p + kx_i
				ok := iy >= 0 && iy < c.it && ix >= 0 && ix < c.ifr
				xm[t * TILE + j] = ok ? c.x[(iy * c.ifr) + ix] : 0
			}
		}
		for oc in 0 ..< c.ocs {
			wrow := c.w[oc * taps:][:taps]
			b := c.bias == nil ? 0 : c.bias[oc]
			dst := outp + uintptr((oc * c.pos + t0) * 4)
			j := 0
			for ; j + 8 <= n; j += 8 {
				acc := simd.f32x8(0)
				for t in 0 ..< taps {
					acc = simd.fma(simd.f32x8(wrow[t]), qk_load8(rawptr(xmp + uintptr((t * TILE + j) * 4))), acc)
				}
				qk_store8(rawptr(dst + uintptr(j * 4)), acc + simd.f32x8(b))
			}
			for ; j < n; j += 1 {
				// Bias LAST, matching the 8-wide body above (acc + b). See the
				// note in conv_1x1_chunk: seeding with the bias would make this
				// position's rounding depend on the chunk boundary and hence on
				// the thread count.
				s: f32 = 0
				for t in 0 ..< taps { s += wrow[t] * xm[t * TILE + j] }
				c.out[oc * c.pos + t0 + j] = s + b
			}
		}
	}
}

// conv_first_3x3s2 is the encoder's first subsampling layer (1 -> ocs outputs,
// kx by ky taps, stride 2, bias). No relu inside: conv2d stays a pure
// convolution for every dispatch path, the caller applies its own.
conv_first_3x3s2 :: proc(
	x: []f32,
	it, ifr: int,
	w: []f32,
	kx, ky, ocs: int,
	bias: []f32,
	s, p: int,
	pool: ^Pool,
	workers: int,
	out: []f32, // [ocs][ot * ofr]
) {
	assert(kx * ky <= 16, "conv_first_3x3s2: tap scratch is 16 deep")
	ot := (it + 2 * p - ky) / s + 1
	ofr := (ifr + 2 * p - kx) / s + 1
	pos := ot * ofr
	ctx := Conv_First_Ctx {
		x    = x,
		w    = w,
		bias = bias,
		out  = out,
		pos  = pos,
		it   = it,
		ot   = ot,
		ofr  = ofr,
		ifr  = ifr,
		kx   = kx,
		ky   = ky,
		s    = s,
		p    = p,
		ocs  = ocs,
	}
	par_chunks(pool, workers, pos, conv_first_chunk, &ctx)
}

Conv_1x1_Ctx :: struct {
	x:    []f32, // [ics][pos]
	w:    []f32, // [ocs][ics]: ggml (1,1,ics,ocs) keeps ics contiguous
	bias: []f32,
	out: []f32, // [ocs][pos]
	pos: int,
	ics: int,
	ocs: int,
}

// conv_1x1_chunk runs eight positions at a time against four output channels at
// once. Four accumulators is what makes it throughput- rather than latency-bound:
// a single fma chain stalls on its own latency (4 cycles per eight products,
// 0.5 MAC/cycle) because every provider of the loaded vector is serial.
@(enable_target_feature = "avx2,fma")
conv_1x1_chunk :: proc(data: rawptr, lo, hi: int) #no_bounds_check {
	c := cast(^Conv_1x1_Ctx)data
	xp := uintptr(raw_data(c.x))
	outp := uintptr(raw_data(c.out))
	ics := c.ics
	pos4 := uintptr(c.pos * 4) // activation row stride, bytes
	w4 := uintptr(ics * 4)    // weight row stride, bytes

	for p0 := lo; p0 < hi; p0 += 8 {
		n := min(8, hi - p0)
		if n != 8 {
			for o in 0 ..< c.ocs {
				row := c.w[o * ics:][:ics]
				for j in 0 ..< n {
					// Bias LAST, matching the 8-wide body below. Seeding the
					// accumulator with the bias instead would push it through
					// every intermediate rounding, so a position landing in this
					// tail would differ in its last bits from the same position
					// handled by the body. Which positions land here depends on
					// per = ceil(pos/nt), which made the output a function of the
					// thread count.
					s: f32 = 0
					for i in 0 ..< ics { s += row[i] * c.x[i * c.pos + p0 + j] }
					c.out[o * c.pos + p0 + j] = s + c.bias[o]
				}
			}
			continue
		}
		o := 0
		for ; o + 4 <= c.ocs; o += 4 {
			a0 := simd.f32x8(0)
			a1 := simd.f32x8(0)
			a2 := simd.f32x8(0)
			a3 := simd.f32x8(0)
			r0 := uintptr(raw_data(c.w[o * ics:][:ics]))
			i := 0
			for ; i < ics; i += 1 {
				xv := qk_load8(rawptr(xp + uintptr(p0 * 4) + uintptr(i) * pos4))
				o4 := uintptr(i * 4)
				a0 = simd.fma(simd.f32x8((cast(^f32)(rawptr(r0 + o4)))^), xv, a0)
				a1 = simd.fma(simd.f32x8((cast(^f32)(rawptr(r0 + w4 + o4)))^), xv, a1)
				a2 = simd.fma(simd.f32x8((cast(^f32)(rawptr(r0 + 2 * w4 + o4)))^), xv, a2)
				a3 = simd.fma(simd.f32x8((cast(^f32)(rawptr(r0 + 3 * w4 + o4)))^), xv, a3)
			}
			dst := outp + uintptr((o * c.pos + p0) * 4)
			qk_store8(rawptr(dst), a0 + simd.f32x8(c.bias[o]))
			qk_store8(rawptr(dst + uintptr(c.pos * 4)), a1 + simd.f32x8(c.bias[o + 1]))
			qk_store8(rawptr(dst + uintptr(c.pos * 8)), a2 + simd.f32x8(c.bias[o + 2]))
			qk_store8(rawptr(dst + uintptr(c.pos * 12)), a3 + simd.f32x8(c.bias[o + 3]))
		}
		for ; o < c.ocs; o += 1 {
			row := c.w[o * ics:][:ics]
			acc := simd.f32x8(0)
			for i in 0 ..< ics {
				acc = simd.fma(simd.f32x8(row[i]), qk_load8(rawptr(xp + uintptr(p0 * 4) + uintptr(i) * pos4)), acc)
			}
			qk_store8(rawptr(outp + uintptr((o * c.pos + p0) * 4)), acc + simd.f32x8(c.bias[o]))
		}
	}
}

// conv_1x1 is a [ics][pos] x [ics][ocs] matmul with a bias, written back
// channel-major. Both the weight rows and the eight-position activation vectors
// are contiguous in that layout, so nothing is transposed.
conv_1x1 :: proc(
	x: []f32,
	ics, pos: int,
	w: []f32, // [ocs][ics]
	ocs: int,
	bias: []f32,
	pool: ^Pool,
	workers: int,
	out: []f32,
) {
	ctx := Conv_1x1_Ctx{x = x, w = w, bias = bias, out = out, pos = pos, ics = ics, ocs = ocs}
	par_chunks(pool, workers, pos, conv_1x1_chunk, &ctx)
}
