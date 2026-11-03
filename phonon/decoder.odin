package phonon

// TDT decoder: a 2-layer LSTM predictor over token embeddings, a joint network
// that fuses the encoder frame with the predictor state, and the greedy
// duration-based loop. Weights live in the GGUF as pred.*, joint.*.

import "core:fmt"
import "core:math"
import "core:mem"

Decoder :: struct {
	model:        ^Gguf,
	vocab:        int, // 8193 = 8192 symbols + the blank row
	blank:        int, // 8192
	pred_dim:     int, // 640
	n_layers:     int, // LSTM layers
	n_dur:        int, // 5 duration bins
	max_symbols:  int,
	dur:          [5]i32,

	last:   int, // token whose embedding sits in `emb`
	emb:    []f32, // [pred_dim] embedding of `last`
	h:      [2][]f32,
	c:      [2][]f32,
	gates:  []f32, // [4 * pred_dim] scratch
	g:      []f32, // [pred_dim] predictor state fed to the joint
	joint:  []f32, // [pred_dim] joint hidden
	logits: []f32, // [vocab + n_dur]
	trace:  Trace, // parity hook, nil in production
	steps:  int,
}

decoder_init :: proc(d: ^Decoder, g: ^Gguf, alloc := context.allocator) {
	d.model = g
	d.vocab = int(gguf_int(g, "stt.parakeet.predictor.vocab", 8193))
	d.blank = int(gguf_int(g, "tokenizer.ggml.blank_token_id", 8192))
	d.pred_dim = int(gguf_int(g, "stt.parakeet.predictor.hidden", 640))
	d.n_layers = int(gguf_int(g, "stt.parakeet.predictor.n_layers", 2))
	d.n_dur = int(gguf_int(g, "stt.parakeet.joint.num_extra_outputs", 5))
	d.max_symbols = int(gguf_int(g, "stt.parakeet.tdt.max_symbols", 10))
	nd := d.pred_dim
	d.emb = make([]f32, nd, alloc)
	d.gates = make([]f32, 4 * nd, alloc)
	d.g = make([]f32, nd, alloc)
	d.joint = make([]f32, nd, alloc)
	for l in 0 ..< 2 {
		d.h[l] = make([]f32, nd, alloc)
		d.c[l] = make([]f32, nd, alloc)
	}
	d.logits = make([]f32, d.vocab + d.n_dur, alloc)
	for i in 0 ..< 5 { d.dur[i] = i32(i) }
	if v, ok := g.kv["stt.parakeet.tdt.durations"]; ok {
		if arr, is_arr := v.([]i64); is_arr {
			for i in 0 ..< min(len(arr), 5) { d.dur[i] = i32(arr[i]) }
		}
	}
}

decoder_destroy :: proc(d: ^Decoder, alloc := context.allocator) {
	delete(d.emb, alloc)
	delete(d.gates, alloc)
	delete(d.joint, alloc)
	delete(d.logits, alloc)
	for l in 0 ..< 2 {
		delete(d.h[l], alloc)
		delete(d.c[l], alloc)
	}
	d^ = {}
}

decoder_reset :: proc(d: ^Decoder) {
	for l in 0 ..< 2 {
		for &v in d.h[l] { v = 0 }
		for &v in d.c[l] { v = 0 }
	}
	for &v in d.g { v = 0 }
	for &v in d.emb { v = 0 }
	d.last = -1
}

// embed decodes one embedding row of pred.embed.weight ([pred_dim, vocab]).
embed_row :: proc(d: ^Decoder, tok: int) {
	w := wt(d.model, "pred.embed.weight")
	row_bytes := len(w.bytes) / d.vocab
	if tok < 0 || tok >= d.vocab {
		for &v in d.emb { v = 0 }
		return
	}
	row := w.bytes[tok * row_bytes:][:row_bytes]
	#partial switch w.type {
	case .Q8_0:
		for b in 0 ..< d.pred_dim / Q8_0_QUANT_SIZEOF {
			blk := row[b * Q8_0_BLOCK_SIZEOF:]
			s := half_to_float(u16(blk[0]) | u16(blk[1]) << 8)
			for i in 0 ..< Q8_0_QUANT_SIZEOF {
				d.emb[b * Q8_0_QUANT_SIZEOF + i] = s * f32(transmute(i8)(blk[2 + i]))
			}
		}
	case .F32:
		copy(d.emb, mem.slice_data_cast([]f32, row[:d.pred_dim * 4]))
	case .F16:
		for i in 0 ..< d.pred_dim {
			d.emb[i] = half_to_float(u16(row[i * 2]) | u16(row[i * 2 + 1]) << 8)
		}
	case:
		panic("unsupported embedding type")
	}
}

// predictor_step runs the LSTM stack for one input vector. The new state is
// left in d.h / d.c and the last layer's hidden state in d.model, which is what
// the joint consumes.
predictor_step :: proc(d: ^Decoder, x_in: []f32) {
	nd := d.pred_dim
	x := x_in
	for l in 0 ..< d.n_layers {
		name := fmt.tprintf("pred.lstm.%d", l)
		wx := wt(d.model, fmt.tprintf("%s.Wx", name))
		wh := wt(d.model, fmt.tprintf("%s.Wh", name))
		bias := fb(d.model, fmt.tprintf("%s.bias", name))
		n_out := 4 * nd
		row_x := len(wx.bytes) / n_out
		row_h := len(wh.bytes) / n_out
		ax := act_of(x, wx.type)
		ah := act_of(d.h[l], wh.type)
		for o in 0 ..< n_out {
			acc := bias[o]
			acc += weight_dot(wx.bytes[o * row_x:][:row_x], scales_row(wx, o), x, wx.type, ax)
			acc += weight_dot(wh.bytes[o * row_h:][:row_h], scales_row(wh, o), d.h[l], wh.type, ah)
			d.gates[o] = acc
		}
		c := d.c[l]
		h := d.h[l]
		for i in 0 ..< nd {
			ig := sigmoid(d.gates[i])
			fg := sigmoid(d.gates[nd + i])
			gg := math.tanh(d.gates[2 * nd + i])
			og := sigmoid(d.gates[3 * nd + i])
			c[i] = fg * c[i] + ig * gg
			h[i] = og * math.tanh(c[i])
		}
		x = h
	}
	copy(d.g, x)
}

// predictor_token embeds `tok` and advances the predictor by it.
predictor_token :: proc(d: ^Decoder, tok: int) {
	embed_row(d, tok)
	d.last = tok
	predictor_step(d, d.emb)
}

// joint_frame writes the joint logits for encoder frame enc_row.
joint_forward :: proc(d: ^Decoder, enc_row: []f32) {
	nd := d.pred_dim
	copy(d.joint, fb(d.model, "joint.enc.bias"))
	{
		w := wt(d.model, "joint.enc.weight")
		row_bytes := len(w.bytes) / nd
		a := act_of(enc_row, w.type)
		for o in 0 ..< nd {
			d.joint[o] += weight_dot(w.bytes[o * row_bytes:][:row_bytes], scales_row(w, o), enc_row, w.type, a)
		}
	}
	{
		w := wt(d.model, "joint.pred.weight")
		b := fb(d.model, "joint.pred.bias")
		row_bytes := len(w.bytes) / nd
		a := act_of(d.g, w.type)
		for o in 0 ..< nd {
			d.joint[o] += b[o] + weight_dot(w.bytes[o * row_bytes:][:row_bytes], scales_row(w, o), d.g, w.type, a)
		}
	}
	for &v in d.joint { v = max(v, 0) } // relu
	{
		w := wt(d.model, "joint.out.weight")
		b := fb(d.model, "joint.out.bias")
		row_bytes := len(w.bytes) / len(d.logits)
		a := act_of(d.joint, w.type)
		for o in 0 ..< len(d.logits) {
			d.logits[o] = b[o] + weight_dot(w.bytes[o * row_bytes:][:row_bytes], scales_row(w, o), d.joint, w.type, a)
		}
	}
}

argmax :: proc(v: []f32) -> int {
	best, bi := v[0], 0
	for x, i in v { if x > best { best, bi = x, i } }
	return bi
}

// decode greedily walks the encoder frames and appends the emitted token ids.
// Mirrors the reference loop: blank advances by the predicted duration, an
// emitted symbol keeps the frame unless the duration advances it, and the
// predictor only advances on emission.
decode :: proc(d: ^Decoder, enc: Mat, out: ^[dynamic]int) {
	decoder_reset(d)
	n_dur := d.n_dur
	dur_base := len(d.logits) - n_dur
	zeros := make([]f32, d.pred_dim, context.temp_allocator)
	predictor_step(d, zeros)
	d.steps = 0

	t := 0
	symbols := 0
	for t < enc.rows {
		joint_forward(d, mat_row(enc, t))
		if d.trace != nil {
			d.trace(fmt.tprintf("dec.joint.%d", d.steps), Mat{rows = 1, cols = len(d.logits), data = d.logits})
			for l in 0 ..< d.n_layers {
				d.trace(fmt.tprintf("dec.lstm.%d.h.%d", l, d.steps), Mat{rows = 1, cols = d.pred_dim, data = d.h[l]})
				d.trace(fmt.tprintf("dec.lstm.%d.c.%d", l, d.steps), Mat{rows = 1, cols = d.pred_dim, data = d.c[l]})
			}
		}
		d.steps += 1
		tok := argmax(d.logits[:dur_base])
		di := argmax(d.logits[dur_base:])
		dur := int(d.dur[di])
		emit := tok != d.blank
		if emit {
			append(out, tok)
			symbols += 1
			if dur == 0 && symbols >= d.max_symbols { dur = 1 }
		} else if dur == 0 {
			dur = 1
		}
		if dur > 0 { symbols = 0 }
		t += dur
		if emit { predictor_token(d, tok) }
	}
}
