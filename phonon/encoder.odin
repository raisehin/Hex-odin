package phonon

// Parakeet encoder: NeMo-style subsampling, then 24 conformer blocks
// (macaron FFN at 0.5, self-attention with Transformer-XL relative positions,
// convolution module with batch norm folded into the depthwise kernel).
//
// Activations are [time][d_model] row-major everywhere; only the 2-D
// subsampling convolutions need an explicit [time][freq] plane.

import "core:fmt"
import "core:math"
import "core:mem"
import "core:thread"
import "core:time"

Encoder_Config :: struct {
	n_layers:  int,
	d_model:   int,
	n_heads:   int,
	d_ff:      int,
	conv_kernel: int,
	sub_channels: int,
	sub_factor:   int,
}

// load_encoder_config reads the architecture from the GGUF metadata.
load_encoder_config :: proc(g: ^Gguf) -> Encoder_Config {
	return Encoder_Config{
		n_layers     = int(gguf_int(g, "stt.parakeet.encoder.n_layers", 24)),
		d_model      = int(gguf_int(g, "stt.parakeet.encoder.d_model", 1024)),
		n_heads      = int(gguf_int(g, "stt.parakeet.encoder.n_heads", 8)),
		d_ff         = int(gguf_int(g, "stt.parakeet.encoder.d_ff", 4096)),
		conv_kernel  = int(gguf_int(g, "stt.parakeet.encoder.conv_kernel", 9)),
		sub_channels = int(gguf_int(g, "stt.parakeet.encoder.subsampling_channels", 256)),
		sub_factor   = int(gguf_int(g, "stt.parakeet.encoder.subsampling_factor", 8)),
	}
}

wt :: proc(g: ^Gguf, name: string) -> Gguf_Tensor {
	t, ok := gguf_tensor(g, name)
	if !ok {
		panic(fmt.tprintf("missing tensor: %s", name))
	}
	return t
}

// fb reads an f32 weight or bias tensor as a plain slice.
fb :: proc(g: ^Gguf, name: string) -> []f32 {
	t := wt(g, name)
	return mem.slice_data_cast([]f32, t.bytes[:len(t.bytes)])
}

// --- 2-D subsampling convolutions -----------------------------------------

// conv2d runs a (3x3 or 1x1) convolution with stride `s` and padding `p` over
// a [time][freq] plane. Weight layout is ggml [kx, ky, in, out] with kx
// contiguous; `x` and `out` are [c][time][freq].
// conv2d is the subsampling convolution: [ic][t][f] input, [oc][t][f] output,
// weights laid out [kx][ky][ic_per_group][oc]. Three shapes cover the model:
// 3x3/1->256 (im2col into a matmul), depthwise 3x3 stride 2, and 1x1 256->256
// (a matmul over the frame axis). Everything dispatches to the same f32
// kernels the linear layers use.
conv2d :: proc(
	x: []f32, ics, it, ifr: int,
	w: []f32, kx, ky, ocs, groups: int,
	bias: []f32,
	s, p: int,
	pool: ^Pool, workers: int,
) -> (out: []f32, ot: int, ofr: int) {
	ot = (it + 2 * p - ky) / s + 1
	ofr = (ifr + 2 * p - kx) / s + 1
	pos := ot * ofr
	out = make([]f32, ocs * pos)
	switch {
	case groups == 1 && kx == 1 && ky == 1 && s == 1 && p == 0:
		conv_1x1(x, ics, pos, w, ocs, bias, pool, workers, out)
	case groups == 1 && ics == 1:
		conv_first_3x3s2(x, it, ifr, w, kx, ky, ocs, bias, s, p, pool, workers, out)
	case groups == ics && groups == ocs:
		// depthwise: one 3x3 kernel per channel, channel planes contiguous
		in_per_group := ics / groups
		for c in 0 ..< ocs {
			b := bias == nil ? 0 : bias[c]
			plane := out[c * pos:][:pos]
			src := x[c * it * ifr:][:it * ifr]
			wc := w[c * kx * ky:][:kx * ky]
			for ty in 0 ..< ot {
				for fx in 0 ..< ofr {
					acc := b
					for ky_i in 0 ..< ky {
						iy := ty * s - p + ky_i
						if iy < 0 || iy >= it { continue }
						for kx_i in 0 ..< kx {
							ix := fx * s - p + kx_i
							if ix < 0 || ix >= ifr { continue }
							acc += wc[ky_i * kx + kx_i] * src[iy * ifr + ix]
						}
					}
					plane[ty * ofr + fx] = acc
				}
			}
			_ = in_per_group
		}
	case:
		panic("conv2d: unsupported shape")
	}
	return
}

relu_in_place :: proc(x: []f32) {
	for &v in x { v = max(v, 0) }
}

// subsampling_conv applies the ggml weight of one pre-encode conv layer and
// adds its bias, mapping the tensor into conv2d's arguments.
sub_conv :: proc(
	e: ^Encoder,
	x: []f32, ics, it, ifr: int,
	name: string,
	groups, s, p: int,
) -> (out: []f32, ot: int, ofr: int) {
	w := wt(e.g, fmt.tprintf("%s.weight", name))
	b := fb(e.g, fmt.tprintf("%s.bias", name))
	kx := int(w.dims[0])
	ky := int(w.dims[1])
	ocs: int
	if w.n_dims > 3 { ocs = int(w.dims[3]) } else { ocs = int(w.dims[2]) }
	ws := mem.slice_data_cast([]f32, w.bytes)
	return conv2d(x, ics, it, ifr, ws, kx, ky, ocs, groups, b, s, p, e.pool, e.workers)
}

// --- relative position embeddings -----------------------------------------

// rel_pos_emb builds the interleaved sin/cos table used by the Transformer-XL
// relative attention: row k holds the embedding for position (T-1)-k.
rel_pos_emb :: proc(T, d_model: int, alloc := context.allocator) -> Mat {
	pe := mat_make(2 * T - 1, d_model, alloc)
	fill_pos_emb(pe)
	return pe
}

// fill_pos_emb writes the table for a [2T-1][d_model] buffer whose row count
// fixes T = (rows + 1) / 2. The div_term vector is hoisted out of the row loop:
// the old form evaluated math.pow once per element (13 824 per block, 24 blocks
// per utterance) for 512 distinct values.
fill_pos_emb :: proc(pe: Mat) {
	T := (pe.rows + 1) / 2
	half := pe.cols / 2
	inv := make([]f32, half, context.temp_allocator)
	for i in 0 ..< half {
		inv[i] = f32(1) / math.pow(f32(10000), f32(2 * i) / f32(pe.cols))
	}
	for k in 0 ..< pe.rows {
		pos := f32(T - 1 - k)
		row := mat_row(pe, k)
		for i in 0 ..< half {
			a := pos * inv[i]
			row[2 * i] = math.sin(a)
			row[2 * i + 1] = math.cos(a)
		}
	}
}

// Trace receives a named [rows][cols] f32 block.
Trace :: #type proc(name: string, m: Mat)

// --- encoder ---------------------------------------------------------------

// Block_Weights holds every tensor one conformer block reads, resolved once in
// encoder_init. The previous code formatted the tensor name and probed the GGUF
// map inside the per-frame layer-norm loop, i.e. six tprintf allocations and
// six map lookups per frame per block (~3 100 heap allocations and as many
// hash probes per utterance) for data that is fixed after load.
Block_Weights :: struct {
	norm_ff1_w, norm_ff1_b: []f32,
	ff1_l1, ff1_l2:         Gguf_Tensor,
	norm_attn_w, norm_attn_b: []f32,
	attn_q, attn_k, attn_v:   Gguf_Tensor,
	attn_pos, attn_out:       Gguf_Tensor,
	pos_bias_u, pos_bias_v:   []f32,
	norm_conv_w, norm_conv_b: []f32,
	conv_pw1, conv_pw2:       Gguf_Tensor,
	bn_w, bn_b:               []f32,
	bn_mean, bn_var:          []f32,
	dw:                       []f32,
	norm_ff2_w, norm_ff2_b:   []f32,
	ff2_l1, ff2_l2:           Gguf_Tensor,
	norm_out_w, norm_out_b:   []f32,
}

// block_wt and block_fb format "<prefix>.<suffix>[.weight|.bias]" once, at load.
@(private = "file")
block_wt :: proc(g: ^Gguf, prefix, suffix: string) -> Gguf_Tensor {
	return wt(g, fmt.tprintf("%s.%s.weight", prefix, suffix))
}

@(private = "file")
block_fb :: proc(g: ^Gguf, prefix, suffix: string) -> []f32 {
	return fb(g, fmt.tprintf("%s.%s", prefix, suffix))
}

// block_weights resolves one block's tensors. The name formats here are the
// only ones left in the encoder's hot path, and this runs once per block at
// load.
block_weights :: proc(g: ^Gguf, layer: int) -> (bw: Block_Weights) {
	prefix := fmt.tprintf("enc.blocks.%d", layer)
	bw = Block_Weights {
		norm_ff1_w   = block_fb(g, prefix, "norm_ff1.weight"),
		norm_ff1_b   = block_fb(g, prefix, "norm_ff1.bias"),
		ff1_l1       = block_wt(g, prefix, "ff1.linear1"),
		ff1_l2       = block_wt(g, prefix, "ff1.linear2"),
		norm_attn_w  = block_fb(g, prefix, "norm_attn.weight"),
		norm_attn_b  = block_fb(g, prefix, "norm_attn.bias"),
		attn_q       = block_wt(g, prefix, "attn.linear_q"),
		attn_k       = block_wt(g, prefix, "attn.linear_k"),
		attn_v       = block_wt(g, prefix, "attn.linear_v"),
		attn_pos     = block_wt(g, prefix, "attn.linear_pos"),
		attn_out     = block_wt(g, prefix, "attn.linear_out"),
		pos_bias_u   = block_fb(g, prefix, "attn.pos_bias_u"),
		pos_bias_v   = block_fb(g, prefix, "attn.pos_bias_v"),
		norm_conv_w  = block_fb(g, prefix, "norm_conv.weight"),
		norm_conv_b  = block_fb(g, prefix, "norm_conv.bias"),
		conv_pw1     = block_wt(g, prefix, "conv.pointwise1"),
		conv_pw2     = block_wt(g, prefix, "conv.pointwise2"),
		bn_w         = block_fb(g, prefix, "conv.bn.weight"),
		bn_b         = block_fb(g, prefix, "conv.bn.bias"),
		bn_mean      = block_fb(g, prefix, "conv.bn.running_mean"),
		bn_var       = block_fb(g, prefix, "conv.bn.running_var"),
		dw           = block_fb(g, prefix, "conv.depthwise.weight"),
		norm_ff2_w   = block_fb(g, prefix, "norm_ff2.weight"),
		norm_ff2_b   = block_fb(g, prefix, "norm_ff2.bias"),
		ff2_l1       = block_wt(g, prefix, "ff2.linear1"),
		ff2_l2       = block_wt(g, prefix, "ff2.linear2"),
		norm_out_w   = block_fb(g, prefix, "norm_out.weight"),
		norm_out_b   = block_fb(g, prefix, "norm_out.bias"),
	}
	return
}

// Encode_Stats is the per-stage split of one encode(), filled only when
// Encoder.stats is non-nil (the bench driver sets it). Marks are two tick_now
// calls per stage per block: 24 blocks x ~14 marks, negligible against 400 ms
// and the only way to see which stage owns the time without guessing.
Encode_Stats :: struct {
	calls:      int,
	total:      f64,
	pre_encode: f64,
	norm:       f64, // the five pre-norm passes + the final out-norm
	ffn:        f64, // both macaron halves, including their SiLU
	silu:       f64, // inside ffn
	attn:       f64,
	attn_qkv:   f64, // linear_q/k/v + the pos projection
	attn_score: f64, // scores, softmax, AV, linear_out
	conv:       f64,
	conv_glu:   f64,
	conv_dw:    f64,
	conv_pw2:   f64,
	resid:      f64, // the four residual adds
	pre_conv:   [8]f64, // pre_encode: the mel scatter, conv.0/2/3/5/6, the stack, the out projection
}

@(private = "file")
ms_since :: proc(t: time.Tick) -> f64 {
	return time.duration_milliseconds(time.tick_since(t))
}

Encoder :: struct {
	g:     ^Gguf,
	cfg:   Encoder_Config,
	// bw is the resolved per-block tensor table: [n_layers]Block_Weights.
	bw:    []Block_Weights,
	// stats, when non-nil, accumulates the per-stage split of encode().
	stats: ^Encode_Stats,
	// alloc owns the per-utterance workspace. It must not be the caller's
	// temp allocator: the workspace outlives encode by design (a second
	// utterance of the same length reuses it) and encoder_destroy frees it.
	alloc: mem.Allocator,
	pool:     ^Pool,
	workers:  int,
	T:     int, // encoder frames
	// trace, when set, is called with every named stage tensor; the parity
	// harness uses it to diff against reference dumps. Nil in production.
	trace: Trace,
	ws:    struct {
		// per-block scratch, allocated once for the utterance
		a: Mat, // [T, d_model] attention/ffn activation
		b: Mat, // [T, d_model]
		ff: Mat, // [T, d_ff]
		q: Mat,
		k: Mat,
		v: Mat,
		p: Mat, // [2T-1, d_model] attention position projection output
		pe: Mat, // [2T-1, d_model] Transformer-XL sin/cos table, T-dependent
		conv: Mat, // [T, 2*d_model] pointwise1
		c: Mat, // [T, d_model] GLU output; never aliases the source
		dw: Mat, // [T, d_model] depthwise+SiLU output; separate buffer: the
		// depthwise pass reads a window of the GLU output per channel, so it
		// must never write where it still reads
		scores: []f32, // [n_heads][T][T]
	},
}

encoder_init :: proc(e: ^Encoder, g: ^Gguf, pool: ^Pool, alloc := context.allocator) {
	e.g = g
	e.cfg = load_encoder_config(g)
	e.pool = pool
	e.workers = pool == nil ? 1 : pool.nt
	e.alloc = alloc
	e.T = -1
	e.bw = make([]Block_Weights, e.cfg.n_layers, alloc)
	for l in 0 ..< e.cfg.n_layers { e.bw[l] = block_weights(g, l) }
	encoder_resize(e, 0)
}

encoder_resize :: proc(e: ^Encoder, T: int) {
	if e.T == T { return }
	e.T = T
	alloc := e.alloc
	d := e.cfg.d_model
	mat_destroy(&e.ws.a, alloc)
	mat_destroy(&e.ws.b, alloc)
	mat_destroy(&e.ws.ff, alloc)
	mat_destroy(&e.ws.q, alloc)
	mat_destroy(&e.ws.k, alloc)
	mat_destroy(&e.ws.v, alloc)
	mat_destroy(&e.ws.p, alloc)
	mat_destroy(&e.ws.pe, alloc)
	mat_destroy(&e.ws.conv, alloc)
	mat_destroy(&e.ws.c, alloc)
	mat_destroy(&e.ws.dw, alloc)
	delete(e.ws.scores, alloc)
	e.ws.a = mat_make(T, d, alloc)
	e.ws.b = mat_make(T, d, alloc)
	e.ws.ff = mat_make(T, e.cfg.d_ff, alloc)
	e.ws.q = mat_make(T, d, alloc)
	e.ws.k = mat_make(T, d, alloc)
	e.ws.v = mat_make(T, d, alloc)
	e.ws.p = mat_make(max(2 * T - 1, 0), d, alloc)
	// The sin/cos table depends only on T and d_model, so it is built once per
	// geometry instead of once per block per utterance (24 x 13 824 math.pow
	// calls before this).
	e.ws.pe = mat_make(max(2 * T - 1, 0), d, alloc)
	if T > 0 { fill_pos_emb(e.ws.pe) }
	e.ws.conv = mat_make(T, 2 * d, alloc)
	e.ws.c = mat_make(T, d, alloc)
	e.ws.dw = mat_make(T, d, alloc)
	e.ws.scores = make([]f32, e.cfg.n_heads * max(T, 0) * max(T, 0), alloc)
}

encoder_destroy :: proc(e: ^Encoder) {
	alloc := e.alloc
	mat_destroy(&e.ws.a, alloc)
	mat_destroy(&e.ws.b, alloc)
	mat_destroy(&e.ws.ff, alloc)
	mat_destroy(&e.ws.q, alloc)
	mat_destroy(&e.ws.k, alloc)
	mat_destroy(&e.ws.v, alloc)
	mat_destroy(&e.ws.p, alloc)
	mat_destroy(&e.ws.pe, alloc)
	mat_destroy(&e.ws.conv, alloc)
	mat_destroy(&e.ws.c, alloc)
	mat_destroy(&e.ws.dw, alloc)
	delete(e.ws.scores, alloc)
	delete(e.bw, alloc)
	e^ = {}
}

// pre_mark closes one pre_encode stage into stats.pre_conv; with stats off it
// costs nothing, not even the tick.
pre_mark :: proc(e: ^Encoder, i: int, t0: ^time.Tick) {
	if e.stats == nil { return }
	e.stats.pre_conv[i] += ms_since(t0^)
	t0^ = time.tick_now()
}

// pre_encode consumes the mel plane [n_mels][frames] and produces [T][d_model].
pre_encode :: proc(e: ^Encoder, mel: []f32, n_mels, frames: int, alloc := context.allocator) -> Mat {
	cfg := e.cfg
	g := e.g
	// ggml conv layout is [kx = freq, ky = time]; the mel arrives as
	// [mels][frames], so it is transposed once into a [time][freq] plane.
	x0 := make([]f32, frames * n_mels, context.temp_allocator)
	for f in 0 ..< n_mels {
		for t in 0 ..< frames {
			x0[t * n_mels + f] = mel[f * frames + t]
		}
	}
	_pt: time.Tick
	if e.stats != nil { _pt = time.tick_now() }
	it, ifr := frames, n_mels
	pre_mark(e, 0, &_pt)
	// conv.0: 3x3 stride 2 pad 1, 1 -> 256
	x1, t1, f1 := sub_conv(e, x0, 1, it, ifr, "enc.pre_encode.conv.0", 1, 2, 1)
	relu_in_place(x1)
	// conv.2: depthwise 3x3 stride 2 pad 1
	pre_mark(e, 1, &_pt)
	x2, t2, f2 := sub_conv(e, x1, cfg.sub_channels, t1, f1, "enc.pre_encode.conv.2", cfg.sub_channels, 2, 1)
	// conv.3: pointwise with bias + relu
	pre_mark(e, 2, &_pt)
	x3, t3, f3 := sub_conv(e, x2, cfg.sub_channels, t2, f2, "enc.pre_encode.conv.3", 1, 1, 0)
	relu_in_place(x3)
	// conv.5: depthwise, conv.6: pointwise
	pre_mark(e, 3, &_pt)
	x5, t5, f5 := sub_conv(e, x3, cfg.sub_channels, t3, f3, "enc.pre_encode.conv.5", cfg.sub_channels, 2, 1)
	pre_mark(e, 4, &_pt)
	x6, t6, f6 := sub_conv(e, x5, cfg.sub_channels, t5, f5, "enc.pre_encode.conv.6", 1, 1, 0)
	relu_in_place(x6)
	T := t6
	C := cfg.sub_channels
	// reshape [ch][t][f] -> [t][ch * f]: channel-major, frequency minor
	pre_mark(e, 5, &_pt)
	stacked := make([]f32, T * C * f6, context.temp_allocator)
	for c in 0 ..< C {
		for t in 0 ..< T {
			for f in 0 ..< f6 {
				stacked[t * C * f6 + c * f6 + f] = x6[c * t6 * f6 + t * f6 + f]
			}
		}
	}
	pre_mark(e, 6, &_pt)
	out := mat_make(T, cfg.d_model, alloc)
	lin_rows_par(out, Mat{rows = T, cols = C * f6, data = stacked}, wt(g, "enc.pre_encode.out.weight"), fb(g, "enc.pre_encode.out.bias"), e.pool, e.workers)
	pre_mark(e, 7, &_pt)
	return out
}

// attention runs one Transformer-XL relative-position self-attention.
attention :: proc(e: ^Encoder, layer: int, x: Mat, out: Mat) {
	cfg := e.cfg
	d, H, T := cfg.d_model, cfg.n_heads, x.rows
	hd := d / H
	bw := e.bw[layer]

	t0 := time.tick_now()
	q, k, v, p := e.ws.q, e.ws.k, e.ws.v, e.ws.p
	lin_rows_par(q, x, bw.attn_q, nil, e.pool, e.workers)
	lin_rows_par(k, x, bw.attn_k, nil, e.pool, e.workers)
	lin_rows_par(v, x, bw.attn_v, nil, e.pool, e.workers)

	lin_rows_par(p, e.ws.pe, bw.attn_pos, nil, e.pool, e.workers)

	if e.stats != nil { e.stats.attn_qkv += ms_since(t0) }

	bu := bw.pos_bias_u
	bv := bw.pos_bias_v
	st := Score_Task {
		q = q, k = k, p = p, scores = e.ws.scores,
		bu = bu, bv = bv,
		H = H, T = T, hd = hd,
		scale = f32(1) / math.sqrt(f32(hd)),
	}
	par_chunks(e.pool, e.workers, H * T, score_rows, &st)
	at := AV_Task {scores = e.ws.scores, v = v, out = out, H = H, T = T, hd = hd}
	par_chunks(e.pool, e.workers, T, av_rows, &at)
}

// The attention score and value-mixing phases are O(T^2) with an unvectorised
// f32 reduction in the inner loop (Odin does not enable fast-math, so LLVM
// cannot reassociate the sum), which made them the dominant cost on anything
// longer than a second: 1.0 s of a 3.7 s encoder at T=103, against 0.4 s for
// the three projection matmuls. Both phases are row-separable — a (head, query)
// row owns its scores and its softmax, and a query row owns its output — so the
// pool gets them without changing a single arithmetic order.

@(private = "file")
Score_Task :: struct {
	q, k, p:  Mat,
	scores:   []f32,
	bu, bv:   []f32,
	H, T, hd:        int,
	scale:           f32,
}

@(private = "file")
score_rows :: proc(ctx: rawptr, lo, hi: int) {
	t := cast(^Score_Task)ctx
	H, T, hd := t.H, t.T, t.hd
	tt := T * T
	for r in lo ..< hi {
		h := r / T
		i := r % T
		off := h * hd
		qi := mat_row(t.q, i)[off:][:hd]
		row := t.scores[h * tt + i * T:][:T]
		for j in 0 ..< T {
			kj := mat_row(t.k, j)[off:][:hd]
			ac: f32 = 0
			for dd in 0 ..< hd { ac += (qi[dd] + t.bu[off + dd]) * kj[dd] }
			// position term for index (j + T - 1 - i)
			pj := mat_row(t.p, j + T - 1 - i)[off:][:hd]
			bd: f32 = 0
			for dd in 0 ..< hd { bd += (qi[dd] + t.bv[off + dd]) * pj[dd] }
			row[j] = t.scale * (ac + bd)
		}
		softmax(row)
	}
}

@(private = "file")
AV_Task :: struct {
	scores: []f32,
	v, out: Mat,
	H, T, hd: int,
}

@(private = "file")
av_rows :: proc(ctx: rawptr, lo, hi: int) {
	t := cast(^AV_Task)ctx
	H, T, hd := t.H, t.T, t.hd
	tt := T * T
	for i in lo ..< hi {
		o := mat_row(t.out, i)
		for h in 0 ..< H {
			off := h * hd
			sc := t.scores[h * tt + i * T:][:T]
			for dd in 0 ..< hd { o[off + dd] = 0 }
			for j in 0 ..< T {
				w := sc[j]
				vj := mat_row(t.v, j)[off:][:hd]
				for dd in 0 ..< hd { o[off + dd] += w * vj[dd] }
			}
		}
	}
}

// conv_module is the conformer convolution: pointwise GLU, depthwise conv with
// the batch norm folded in, SiLU, pointwise out.
// The GLU and the depthwise pass each read a per-frame or per-channel window
// and write a different buffer, so they split cleanly over the pool: one row
// (frame) for the GLU, one channel's whole time line for the convolution.
@(private = "file")
Glu_Ctx :: struct {
	hw, a: Mat,
	d:     int,
}

@(private = "file")
glu_chunk :: proc(p: rawptr, lo, hi: int) {
	c := cast(^Glu_Ctx)p
	for i in lo ..< hi {
		hrow := mat_row(c.hw, i)
		arow := mat_row(c.a, i)
		for j in 0 ..< c.d {
			arow[j] = hrow[j] * sigmoid(hrow[c.d + j])
		}
	}
}

@(private = "file")
Depthwise_Ctx :: struct {
	a, b:          Mat,
	dw:            []f32,
	bn_w, bn_b:    []f32,
	mean, var_:    []f32,
	d, T, K:       int,
}

@(private = "file")
depthwise_chunk :: proc(p: rawptr, lo, hi: int) {
	c := cast(^Depthwise_Ctx)p
	half := c.K / 2
	for ch in lo ..< hi {
		// y = x * sc + shift with shift = bias - mean * sc (transcribe.cpp
		// fuses BatchNorm the same way; multiplying bias by sc as well is a
		// 6.8% error at block 0 on the reference dumps).
		sc := c.bn_w[ch] / math.sqrt(c.var_[ch] + 1e-5)
		bc := c.bn_b[ch] - c.mean[ch] * sc
		for t in 0 ..< c.T {
			acc := bc
			for k in 0 ..< c.K {
				ti := t + k - half
				if ti < 0 || ti >= c.T { continue }
				acc += c.dw[ch * c.K + k] * sc * c.a.data[ti * c.d + ch]
			}
			c.b.data[t * c.d + ch] = silu(acc)
		}
	}
}

conv_module :: proc(e: ^Encoder, layer: int, x: Mat, out: Mat) {
	cfg := e.cfg
	d, T, K := cfg.d_model, x.rows, cfg.conv_kernel
	bw := e.bw[layer]

	t0 := time.tick_now()
	hw := e.ws.conv
	lin_rows_par(hw, x, bw.conv_pw1, nil, e.pool, e.workers)
	if e.stats != nil { e.stats.conv += ms_since(t0) }
	t0 = time.tick_now()
	a := e.ws.c
	{
		glu := Glu_Ctx{hw = hw, a = a, d = d}
		par_chunks(e.pool, e.workers, T, glu_chunk, &glu)
	}

	bn_w := bw.bn_w
	bn_b := bw.bn_b
	mean := bw.bn_mean
	var_ := bw.bn_var
	dw := bw.dw // ggml [K, 1, d]: channel-major, K contiguous

	// The depthwise pass reads a K-wide window of the GLU output per channel,
	// so its destination has to be its own buffer: writing over the source
	// makes each frame read values the same loop already replaced.
	if e.stats != nil { e.stats.conv_glu += ms_since(t0) }
	t0 = time.tick_now()
	b := e.ws.dw
	{
		dwc := Depthwise_Ctx {
			a    = a,
			b    = b,
			dw   = dw,
			bn_w = bn_w,
			bn_b = bn_b,
			mean = mean,
			var_ = var_,
			d    = d,
			T    = T,
			K    = K,
		}
		par_chunks(e.pool, e.workers, d, depthwise_chunk, &dwc)
	}
	if e.stats != nil { e.stats.conv_dw += ms_since(t0) }
	t0 = time.tick_now()
	lin_rows_par(out, b, bw.conv_pw2, nil, e.pool, e.workers)
	if e.stats != nil { e.stats.conv_pw2 += ms_since(t0) }
}

// ffn is W2 * silu(W1 * x).
ffn :: proc(e: ^Encoder, w1, w2: Gguf_Tensor, x: Mat, out: Mat) {
	ff := e.ws.ff
	t0 := time.tick_now()
	lin_rows_par(ff, x, w1, nil, e.pool, e.workers)
	if e.stats != nil { e.stats.ffn += ms_since(t0) }
	t0 = time.tick_now()
	silu_par(ff, e.pool, e.workers)
	if e.stats != nil { e.stats.silu += ms_since(t0) }
	t0 = time.tick_now()
	lin_rows_par(out, ff, w2, nil, e.pool, e.workers)
	if e.stats != nil { e.stats.ffn += ms_since(t0) }
}

block :: proc(e: ^Encoder, layer: int, x: Mat) {
	bw := e.bw[layer]
	a, b := e.ws.a, e.ws.b

	// Every per-frame tensor name in this proc used to be formatted and looked
	// up inside the frame loop; bw carries all of them, resolved at load.
	t0 := time.tick_now()
	// macaron ff1 at 0.5
	norm_mat(a, x, bw.norm_ff1_w, bw.norm_ff1_b)
	if e.stats != nil { e.stats.norm += ms_since(t0) }
	ffn(e, bw.ff1_l1, bw.ff1_l2, a, b)
	t0 = time.tick_now()
	add_scaled(x, b, 0.5)
	if e.stats != nil { e.stats.resid += ms_since(t0) }
	if e.trace != nil { e.trace(fmt.tprintf("enc.block.%d.ff1", layer), x) }

	// self-attention
	t0 = time.tick_now()
	norm_mat(a, x, bw.norm_attn_w, bw.norm_attn_b)
	if e.stats != nil { e.stats.norm += ms_since(t0) }
	t0 = time.tick_now()
	attention(e, layer, a, b)
	lin_rows_par(a, b, bw.attn_out, nil, e.pool, e.workers)
	if e.stats != nil { e.stats.attn_score += ms_since(t0) }
	t0 = time.tick_now()
	add_scaled(x, a, 1.0)
	if e.stats != nil { e.stats.resid += ms_since(t0) }
	if e.trace != nil { e.trace(fmt.tprintf("enc.block.%d.attn", layer), x) }

	// convolution module
	t0 = time.tick_now()
	norm_mat(a, x, bw.norm_conv_w, bw.norm_conv_b)
	if e.stats != nil { e.stats.norm += ms_since(t0) }
	conv_module(e, layer, a, b)
	t0 = time.tick_now()
	add_scaled(x, b, 1.0)
	if e.stats != nil { e.stats.resid += ms_since(t0) }
	if e.trace != nil { e.trace(fmt.tprintf("enc.block.%d.conv", layer), x) }

	// macaron ff2 at 0.5
	t0 = time.tick_now()
	norm_mat(a, x, bw.norm_ff2_w, bw.norm_ff2_b)
	if e.stats != nil { e.stats.norm += ms_since(t0) }
	ffn(e, bw.ff2_l1, bw.ff2_l2, a, b)
	t0 = time.tick_now()
	add_scaled(x, b, 0.5)
	if e.stats != nil { e.stats.resid += ms_since(t0) }
	if e.trace != nil { e.trace(fmt.tprintf("enc.block.%d.ff2", layer), x) }

	// out norm
	t0 = time.tick_now()
	layer_norm_rows(x, bw.norm_out_w, bw.norm_out_b)
	if e.stats != nil { e.stats.norm += ms_since(t0) }
}

// encode runs the whole encoder over r-mel features and returns [T][d_model].
encode :: proc(e: ^Encoder, mel: []f32, n_mels, frames: int, alloc := context.allocator) -> Mat {
	t_enc := time.tick_now()
	t0 := time.tick_now()
	if e.stats != nil { e.stats.calls += 1 }
	pre := pre_encode(e, mel, n_mels, frames, alloc)
	if e.stats != nil { e.stats.pre_encode += ms_since(t0) }
	x := pre
	encoder_resize(e, x.rows)
	if e.trace != nil { e.trace("enc.pre_encode.out", x) }
	for l in 0 ..< e.cfg.n_layers {
		block(e, l, x)
	}
	if e.trace != nil { e.trace("enc.final", x) }
	if e.stats != nil { e.stats.total = ms_since(t_enc) }
	return x
}
