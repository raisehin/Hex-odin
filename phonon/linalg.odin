package phonon

// Kernels for the Phonon port: the model file keeps 225 of its 697 tensors as
// Q8_0 blocks and 48 as F16, so every matmul decodes its weight rows on the
// fly. Everything else is f32.
//
// Weight layout is ggml's: a tensor is [ne0 = input, ne1 = output, ...] with
// ne0 contiguous, so row o of a linear weight is the ne0-length vector that
// produces output o. A [ne0, ne1] Q8_0 tensor therefore costs ne0/32 * 34
// bytes per output element.

import "base:intrinsics"
import "core:math"
import "core:mem"
import "core:simd"
import "core:sync"
import "core:thread"

Q8_0_BLOCK_SIZEOF :: 34 // f16 scale + 32 int8, packed
Q8_0_QUANT_SIZEOF :: 32

// Mat is a row-major [rows][cols] f32 block; rows are time frames for
// activations and outputs for weights.
Mat :: struct {
	rows: int,
	cols: int,
	data: []f32,
}

mat_make :: proc(rows, cols: int, alloc := context.allocator) -> Mat {
	return Mat{rows = rows, cols = cols, data = make([]f32, rows * cols, alloc)}
}

mat_destroy :: proc(m: ^Mat, alloc := context.allocator) {
	if m.data != nil { delete(m.data, alloc) }
	m^ = {}
}

mat_row :: proc(m: Mat, i: int) -> []f32 {
	return m.data[i * m.cols:][:m.cols]
}

half_to_float :: #force_inline proc "contextless" (h: u16) -> f32 {
	sign := u32(h & 0x8000) << 16
	exp := u32(h >> 10) & 0x1F
	mant := u32(h & 0x03FF)
	switch {
	case exp == 0x1F:
		return transmute(f32)(sign | 0x7F80_0000 | (mant << 13))
	case exp == 0:
		if mant == 0 { return transmute(f32)(sign) }
		e := u32(127 - 15)
		for mant & 0x0400 == 0 {
			mant <<= 1
			e -= 1
		}
		return transmute(f32)(sign | (e << 23) | ((mant & 0x03FF) << 13))
	}
	return transmute(f32)(sign | ((exp + 112) << 23) | (mant << 13))
}

// act_of quantises x for Q8_0 weights (ggml does the same inside mul_mat);
// other weight types read the f32 activation directly, so it returns {}.
act_of :: proc(x: []f32, t: GGML_Type, alloc := context.temp_allocator) -> Act {
	if t != .Q8_0 { return {} }
	a := act_make(1, len(x), alloc)
	quantize_row_q8_0(x, a.q, a.d)
	return a
}

// scales_row returns the block scales of output row o of a Q8_0 tensor (nil
// for the plain f32 weight blocks the convs build). A row holds ne0 values,
// i.e. ne0 / 32 blocks, each with its own scale: the tensor-wide scales slice
// is [ne1][blocks] row-major, so the count has to come from dims. Deriving it
// from the byte ratio instead yields 1 (rows * blocks * 34 / rows * blocks *
// 34), which silently applies the first block's scale to every block.
scales_row :: proc(t: Gguf_Tensor, o: int) -> []f32 {
	if t.scales == nil { return nil }
	blocks := int(t.dims[0]) / Q8_0_QUANT_SIZEOF
	return t.scales[o * blocks:][:blocks]
}

// weight_dot dispatches on the on-disk type of one weight row. a must be the
// quantised form of x when the weights are Q8_0; ws are that row's block scales.
weight_dot :: proc(w: []u8, ws: []f32, x: []f32, t: GGML_Type, a: Act) -> f32 {
	#partial switch t {
	case .Q8_0: return q8_0_dot_q8(w, ws, a.q, a.d)
	case .F16:  panic("F16 weights are promoted to F32 at load")
	case .F32:  return f32_dot_avx(w, x)
	}
	panic("unsupported weight type")
}

// lin computes out[o] = bias[o] + w_row(o) . x for a [ne0, ne1] weight.
// A nil bias means no bias, as for every attention projection in the encoder.
lin :: proc(out, x: []f32, w: Gguf_Tensor, bias: []f32) {
	alloc := context.temp_allocator
	a := act_of(x, w.type, alloc)
	defer if a.q != nil { act_destroy(&a, alloc) }
	lin_act(out, x, w, bias, a)
}

// lin_act is lin with the activation already quantised (one frame).
// f32_tensor views a plain f32 weight block as a tensor so it can go through
// the same linear kernels as the quantised ones (the subsampling convs use
// this for their 1x1 and im2col matmuls).
f32_tensor :: proc(data: []f32) -> Gguf_Tensor {
	return Gguf_Tensor {type = .F32, bytes = mem.slice_data_cast([]u8, data)}
}

lin_act :: proc(out, x: []f32, w: Gguf_Tensor, bias: []f32, a: Act) {
	rows := len(out)
	row_bytes := len(w.bytes) / rows
	for o in 0 ..< rows {
		row := w.bytes[o * row_bytes:][:row_bytes]
		v := weight_dot(row, scales_row(w, o), x, w.type, a)
		out[o] = bias == nil ? v : v + bias[o]
	}
}

lin_rows :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32) {
	acts := act_make(x.rows, x.cols)
	defer act_destroy(&acts)
	if w.type == .Q8_0 { quantize_acts(acts, x) }
	lin_rows_core(out, x, w, bias, acts, 0, out.cols)
}

// lin_rows_core computes output rows [lo, hi) for every frame. Q8_0 rows are
// taken two at a time through the 2-row x 2-frame kernel, so one pass over the
// weight bytes feeds four dots; the odd row and the odd frame fall back to the
// 1-row kernels. Both the serial and the pooled entry points call this, so a
// worker's rows are computed by exactly the code the serial path runs and the
// two cannot drift.
@(private = "file")
lin_rows_core :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32, acts: Act, lo, hi: int) {
	rows := out.cols
	frames := x.rows
	blocks := act_blocks(x.cols)
	row_bytes := len(w.bytes) / rows
	o := lo
	if w.type == .Q8_0 {
		for ; o + 1 < hi; o += 2 {
			ra := w.bytes[o * row_bytes:][:row_bytes]
			rb := w.bytes[(o + 1) * row_bytes:][:row_bytes]
			wsa := scales_row(w, o)
			wsb := scales_row(w, (o + 1))
			ba := bias == nil ? 0 : bias[o]
			bb := bias == nil ? 0 : bias[(o + 1)]
			has_bias := bias != nil
			f := 0
			for ; f + 1 < frames; f += 2 {
				v00, v10, v01, v11 := q8_0_dot_q8_2f_2r(
					ra, wsa, rb, wsb,
					acts.q[f * x.cols:][:x.cols], acts.d[f * blocks:][:blocks],
					acts.q[(f + 1) * x.cols:][:x.cols], acts.d[(f + 1) * blocks:][:blocks],
				)
				out.data[f * rows + o] = has_bias ? v00 + ba : v00
				out.data[f * rows + o + 1] = has_bias ? v10 + bb : v10
				out.data[(f + 1) * rows + o] = has_bias ? v01 + ba : v01
				out.data[(f + 1) * rows + o + 1] = has_bias ? v11 + bb : v11
			}
			for ; f < frames; f += 1 {
				a := Act {
					q = acts.q[f * x.cols:][:x.cols],
					d = acts.d[f * blocks:][:blocks],
				}
				xa := x.data[f * x.cols:][:x.cols]
				va := weight_dot(ra, wsa, xa, w.type, a)
				vb := weight_dot(rb, wsb, xa, w.type, a)
				out.data[f * rows + o] = has_bias ? va + ba : va
				out.data[f * rows + o + 1] = has_bias ? vb + bb : vb
			}
		}
	}
	for ; o < hi; o += 1 {
		row := w.bytes[o * row_bytes:][:row_bytes]
		ws := scales_row(w, o)
		b := bias == nil ? 0 : bias[o]
		f := 0
		if w.type == .Q8_0 {
			for ; f + 2 <= frames; f += 2 {
				va, vb := q8_0_dot_q8_2f(
					row, ws,
					acts.q[f * x.cols:][:x.cols], acts.d[f * blocks:][:blocks],
					acts.q[(f + 1) * x.cols:][:x.cols], acts.d[(f + 1) * blocks:][:blocks],
				)
				out.data[f * rows + o] = bias == nil ? va : va + b
				out.data[(f + 1) * rows + o] = bias == nil ? vb : vb + b
			}
		}
		for ; f < frames; f += 1 {
			a := Act {
				q = acts.q[f * x.cols:][:x.cols],
				d = acts.d[f * blocks:][:blocks],
			}
			v := weight_dot(row, ws, x.data[f * x.cols:][:x.cols], w.type, a)
			out.data[f * rows + o] = bias == nil ? v : v + b
		}
	}
}

// --- worker pool -----------------------------------------------------------

// Pool dispatches one context to all workers and returns when every worker is
// done. wid 0 is the calling thread; wid > 0 are parked on a semaphore between
// dispatches, so an idle pool costs nothing. The encoder runs hundreds of
// matmuls per utterance, so wake latency must not be paid per task: one
// barrier per matmul, not one per task.
//
// Each worker waits on its own semaphore. A single shared work semaphore looks
// tempting but is wrong: any worker can take a second permit while another is
// descheduled, so one slice runs twice and another never runs, and the straggler
// then writes into buffers the caller has already moved past. That failure shows
// up only at high thread counts (low ones never have spare permits) and looks
// like a handful of corrupted output rows.
Pool :: struct {
	nt:       int,
	threads:  []^thread.Thread,
	args:     []Worker_Arg,
	sem_done: sync.Sema,
	stop:     u32,
	fn:       Task_Proc,
	ctx:      rawptr,
}

Task_Proc :: #type proc(ctx: rawptr, wid: int)

Worker_Arg :: struct {
	p:   ^Pool,
	wid: int,
	sem: sync.Sema, // one permit per dispatch for this worker only
}

pool_make :: proc(nt: int) -> ^Pool {
	if nt < 2 { return nil }
	p := new(Pool)
	p.nt = nt
	p.threads = make([]^thread.Thread, nt)
	p.args = make([]Worker_Arg, nt)
	for i in 1 ..< nt {
		p.args[i] = Worker_Arg{p = p, wid = i}
		p.threads[i] = thread.create_and_start_with_data(&p.args[i], pool_worker)
	}
	return p
}

pool_destroy :: proc(p: ^Pool) {
	if p == nil { return }
	intrinsics.atomic_store(&p.stop, 1)
	for i in 1 ..< p.nt { sync.post(&p.args[i].sem, 1) }
	for i in 1 ..< p.nt {
		if p.threads[i] != nil {
			thread.join(p.threads[i])
			thread.destroy(p.threads[i])
		}
	}
	delete(p.threads)
	delete(p.args)
	free(p)
}

@(private = "file")
pool_worker :: proc(data: rawptr) {
	a := cast(^Worker_Arg)data
	p := a.p
	for {
		sync.wait(&a.sem)
		if intrinsics.atomic_load(&p.stop) != 0 { return }
		p.fn(p.ctx, a.wid)
		sync.post(&p.sem_done, 1)
	}
}

// pool_run runs fn(ctx, wid) on every worker and waits for all of them.
pool_run :: proc(p: ^Pool, fn: Task_Proc, ctx: rawptr) {
	if p == nil { fn(ctx, 0); return }
	p.fn = fn
	p.ctx = ctx
	for i in 1 ..< p.nt { sync.post(&p.args[i].sem, 1) }
	fn(ctx, 0)
	for _ in 0 ..< p.nt - 1 { sync.wait(&p.sem_done) }
}

// par_chunks splits [0, n) over the pool and calls fn on each worker's slice.
// The calling thread takes slice 0, so a nil pool runs the whole range here.
Chunk_Proc :: #type proc(data: rawptr, lo, hi: int)

@(private = "file")
Chunk_Task :: struct {
	data: rawptr,
	fn:   Chunk_Proc,
	n:    int,
	nt:   int,
}

@(private = "file")
chunk_task :: proc(ctx: rawptr, wid: int) {
	t := cast(^Chunk_Task)ctx
	per := (t.n + t.nt - 1) / t.nt
	lo := min(wid * per, t.n)
	hi := min(lo + per, t.n)
	if lo < hi { t.fn(t.data, lo, hi) }
}

par_chunks :: proc(p: ^Pool, workers, n: int, fn: Chunk_Proc, data: rawptr) {
	if n <= 0 { return }
	nt := p == nil ? 1 : min(workers, n)
	if nt < 2 {
		fn(data, 0, n)
		return
	}
	task := Chunk_Task{data = data, fn = fn, n = n, nt = nt}
	pool_run(p, chunk_task, &task)
}

// --- linear layers ---------------------------------------------------------

Lin_Task :: struct {
	out, x: Mat,
	w:      Gguf_Tensor,
	bias:   []f32,
	a:      Act,
	nt:     int,
	// Output-row range this task owns. lin_rows_par covers the whole matrix;
	// lin_rows_par_span covers a caller slice so a GPU+CPU split can share
	// one quantisation.
	lo, hi: int,
}

// lin_task owns an output-row range and runs every frame inside it, so a
// weight row is read once per dispatch instead of once per frame. The
// activations arrive already quantised, one row per frame.
@(private = "file")
lin_task :: proc(ctx: rawptr, wid: int) {
	t := cast(^Lin_Task)ctx
	// Rows are paired inside lin_rows_core, so every worker must start on an
	// even row: a split that begins mid-pair would run the 1-row kernel on a
	// row the serial path pairs, and the two summation trees differ in the
	// last bits. Rounding the stride up to 2 costs at most one row of work.
	// Callers must pass an even lo; mid-splits in the GPU path round down.
	per := (t.hi - t.lo + t.nt - 1) / t.nt
	if per % 2 != 0 { per += 1 }
	lo := min(t.lo + wid * per, t.hi)
	hi := min(lo + per, t.hi)
	if lo < hi { lin_rows_core(t.out, t.x, t.w, t.bias, t.a, lo, hi) }
}

// lin_rows_par spreads the output rows of one linear over the worker pool.
lin_rows_par :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32, pool: ^Pool, workers: int) {
	nt := pool == nil ? 1 : min(workers, out.cols)
	if nt < 2 {
		lin_rows(out, x, w, bias)
		return
	}
	acts := act_make(x.rows, x.cols)
	defer act_destroy(&acts)
	if w.type == .Q8_0 { quantize_acts(acts, x) }
	task := Lin_Task{out = out, x = x, w = w, bias = bias, a = acts, nt = nt, lo = 0, hi = out.cols}
	pool_run(pool, lin_task, &task)
}

// lin_rows_par_span is lin_rows_par for output rows [lo, hi) with caller-owned
// activations, so a GPU+CPU row-split shares one quantisation. lo must be
// even (see lin_task); hi may be anything.
lin_rows_par_span :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32, acts: Act, pool: ^Pool, workers, lo, hi: int) {
	nt := pool == nil ? 1 : min(workers, max(hi - lo, 0))
	if nt < 2 {
		lin_rows_core(out, x, w, bias, acts, lo, hi)
		return
	}
	task := Lin_Task{out = out, x = x, w = w, bias = bias, a = acts, nt = nt, lo = lo, hi = hi}
	pool_run(pool, lin_task, &task)
}

// --- elementwise -----------------------------------------------------------

// silu and sigmoid run per element through math.exp. The kernels below work on
// whole Mats and let LLVM keep the loop arithmetic-free; exp itself stays
// scalar (there is no vector exp, and the encoder's tanh-approx budget does not
// allow a cheaper one: it moves the logits).
@(private = "file")
Silu_Ctx :: struct {
	data: []f32,
}

@(private = "file")
silu_chunk :: proc(p: rawptr, lo, hi: int) {
	c := cast(^Silu_Ctx)p
	silu_in_place(c.data[lo:hi])
}

// silu_par is silu_in_place spread over the pool. The elementwise stages are
// ~2.7 M exp() calls per utterance and were the second largest single-thread
// cost behind the matmuls.
silu_par :: proc(x: Mat, pool: ^Pool, workers: int) {
	ctx := Silu_Ctx{data = x.data}
	par_chunks(pool, workers, len(x.data), silu_chunk, &ctx)
}

@(enable_target_feature = "avx2")
silu_in_place :: proc(x: []f32) {
	for &v in x { v = silu(v) }
}

// norm_mat writes LN(src row) into dst row for every row: the encoder always
// needs the pre-norm activation preserved, so it is never in place.
norm_mat :: proc(dst, src: Mat, gamma, beta: []f32) #no_bounds_check {
	for i in 0 ..< src.rows {
		layer_norm_into(dst.data[i * dst.cols:][:dst.cols], src.data[i * src.cols:][:src.cols], gamma, beta)
	}
}

// layer_norm_rows normalises every row of m in place.
layer_norm_rows :: proc(m: Mat, gamma, beta: []f32) #no_bounds_check {
	for i in 0 ..< m.rows {
		layer_norm(m.data[i * m.cols:][:m.cols], gamma, beta)
	}
}

// add_scaled adds scale * src to dst, both [rows][cols]; the conformer residual
// adds (scale 0.5 for the macaron FFNs, 1 for attention and conv).
@(enable_target_feature = "avx2")
add_scaled :: proc(dst, src: Mat, scale: f32) #no_bounds_check {
	n := len(src.data)
	s := simd.f32x8(scale)
	i := 0
	for ; i + 8 <= n; i += 8 {
		d := transmute(simd.f32x8)((cast(^[8]f32)raw_data(dst.data[i:]))^)
		v := transmute(simd.f32x8)((cast(^[8]f32)raw_data(src.data[i:]))^)
		(cast(^[8]f32)raw_data(dst.data[i:]))^ = transmute([8]f32)simd.fma(v, s, d)
	}
	for ; i < n; i += 1 { dst.data[i] += scale * src.data[i] }
}

silu :: proc "contextless" (x: f32) -> f32 { return x / (1 + math.exp(-x)) }

sigmoid :: proc "contextless" (x: f32) -> f32 { return 1 / (1 + math.exp(-x)) }

softmax :: proc(v: []f32) {
	m := v[0]
	for x in v { m = max(m, x) }
	s: f32 = 0
	for x, i in v {
		v[i] = math.exp(x - m)
		s += v[i]
	}
	inv := f32(1) / s
	for &x in v { x *= inv }
}

// layer_norm_into writes LN(src) into dst; the encoder needs the pre-norm
// activation (x) preserved, so the norm cannot run in place.
layer_norm_into :: proc(dst, src, gamma, beta: []f32, eps := f32(1e-5)) {
	n := f32(len(src))
	mean: f32 = 0
	for x in src { mean += x }
	mean /= n
	v: f32 = 0
	for x in src { d := x - mean; v += d * d }
	inv := f32(1) / math.sqrt(v / n + eps)
	for i in 0 ..< len(src) {
		dst[i] = (src[i] - mean) * inv * gamma[i] + beta[i]
	}
}

// layer_norm normalises one row in place; eps is 1e-5 in this model.
layer_norm :: proc(row, gamma, beta: []f32, eps := f32(1e-5)) {
	n := f32(len(row))
	mean: f32 = 0
	for x in row { mean += x }
	mean /= n
	v: f32 = 0
	for x in row { d := x - mean; v += d * d }
	inv := f32(1) / math.sqrt(v / n + eps)
	for i in 0 ..< len(row) {
		row[i] = (row[i] - mean) * inv * gamma[i] + beta[i]
	}
}
