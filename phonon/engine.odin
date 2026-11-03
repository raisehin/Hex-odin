package phonon

// Engine ties the pieces together: load the GGUF, run the mel frontend, encode,
// decode, detokenise. It owns every allocation for the model and the utterance
// workspace so the caller only deals with text.

import "core:fmt"
import "core:mem"
import "core:strings"
import "core:os"
import "core:thread"
import "core:time"

Engine :: struct {
	g:       ^Gguf,
	enc:     Encoder,
	dec:     Decoder,
	pool:    ^Pool,
	own_pool: bool,
	alloc:   mem.Allocator,
	// vocabulary decoded once at load
	tokens:  []string,
	// F16 tensors promoted to F32 at load (see promote_f16); owned here so
	// the views stay valid for the engine's lifetime.
	promoted: [][]u8,
}

// promote_f16 rewrites every F16 tensor as F32. The encoder reads its weights
// once per utterance on the CPU and the reference does the same promotion for
// its CPU backend; keeping F16 would need a separate F16C kernel path for a
// handful of tensors, while F32 dots run at full width.
promote_f16 :: proc(g: ^Gguf, alloc: mem.Allocator) -> (owned: [][]u8) {
	list := make([dynamic][]u8, 0, 64, alloc)
	for name in g.order {
		t := g.tensors[name] or_continue
		if t.type != .F16 { continue }
		n := len(t.bytes) / 2
		dst := make([]u8, n * 4, alloc)
		src := mem.slice_data_cast([]u16, t.bytes[:n * 2])
		fs := mem.slice_data_cast([]f32, dst)
		for i in 0 ..< n { fs[i] = half_to_float(src[i]) }
		t2 := t
		t2.type = .F32
		t2.bytes = dst
		g.tensors[name] = t2
		append(&list, dst)
	}
	return list[:]
}

// set_trace installs the parity hooks for the encoder and decoder stages.
// set_stats enables the per-stage encode timer for the bench driver. Nil
// (the default) leaves the encoder free of timing calls.
set_stats :: proc(e: ^Engine, s: ^Encode_Stats) {
	e.enc.stats = s
}

set_trace :: proc(e: ^Engine, t: Trace) {
	e.enc.trace = t
	e.dec.trace = t
}

engine_open :: proc(path: string, threads := 0, alloc := context.allocator) -> (e: Engine, err: string) {
	g, load_err := gguf_load(path, alloc)
	if load_err != "" { return {}, load_err }
	pg := new(Gguf, alloc)
	pg^ = g
	e.g = pg
	e.alloc = alloc
	e.promoted = promote_f16(pg, alloc)
	e.pool = thread_pool_make(threads)
	e.own_pool = true
	encoder_init(&e.enc, e.g, e.pool, alloc)
	decoder_init(&e.dec, e.g, alloc)
	if v, ok := e.g.kv["tokenizer.ggml.tokens"]; ok {
		if arr, is_arr := v.([]string); is_arr { e.tokens = arr }
	}
	return
}

engine_close :: proc(e: ^Engine) {
	encoder_destroy(&e.enc)
	decoder_destroy(&e.dec, e.alloc)
	if e.own_pool { thread_pool_free(e.pool) }
	for b in e.promoted { delete(b, e.alloc) }
	delete(e.promoted, e.alloc)
	gguf_destroy(e.g)
		free(e.g, e.alloc)
	e^ = {}
}

// Transcribe returns the decoded text for mono 16 kHz samples.
Transcribe :: struct {
	text:    string,
	tokens:  []int,
	mel_ms:  f64,
	enc_ms:  f64,
	dec_ms:  f64,
}

transcribe :: proc(e: ^Engine, samples: []f32, alloc := context.allocator) -> (res: Transcribe) {
	mark := time.tick_now()
	m: Mel
	mel_init(&m)
	feat := mel_compute(&m, samples)
	defer delete(feat.data, context.allocator)

	// The encoder consumes [mels][frames].
	plane := make([]f32, MEL_N_MELS * feat.frames, context.temp_allocator)
	for t in 0 ..< feat.frames {
		for band in 0 ..< MEL_N_MELS {
			plane[band * feat.frames + t] = feat.data[t * MEL_N_MELS + band]
		}
	}

	if e.enc.trace != nil {
		e.enc.trace("enc.mel.in", Mat{rows = MEL_N_MELS, cols = feat.frames, data = plane})
	}
	res.mel_ms = time.duration_milliseconds(time.tick_since(mark))
	enc := encode(&e.enc, plane, MEL_N_MELS, feat.frames, context.temp_allocator)
	res.enc_ms = time.duration_milliseconds(time.tick_since(mark)) - res.mel_ms
	ids := make([dynamic]int, context.temp_allocator)
	decode(&e.dec, enc, &ids)
	res.dec_ms = time.duration_milliseconds(time.tick_since(mark)) - res.mel_ms - res.enc_ms
	res.tokens = make([]int, len(ids), alloc)
	copy(res.tokens, ids[:])
	res.text = detokenize(e.tokens[:], res.tokens[:], alloc)
	return
}

// detokenize joins SentencePiece pieces: a leading U+2581 maps to a space.
detokenize :: proc(tokens: []string, ids: []int, alloc := context.allocator) -> string {
	b: strings.Builder
	strings.builder_init(&b, alloc)
	for id in ids {
		if id < 0 || id >= len(tokens) { continue }
		piece := tokens[id]
		for r, i in piece {
			if r == '▁' {
				strings.write_byte(&b, ' ')
			} else {
				strings.write_rune(&b, r)
			}
			_ = i
		}
	}
	s := strings.to_string(b)
	return strings.clone(strings.trim_space(s), alloc)
}

thread_pool_make :: proc(threads: int) -> ^Pool {
	n := threads
	if n <= 0 { n = os.get_processor_core_count() }
	return pool_make(n)
}

thread_pool_free :: proc(p: ^Pool) { pool_destroy(p) }

// itoa formats an integer without the fmt machinery per call site.
itoa :: proc(i: int) -> string {
	return fmt.tprintf("%d", i)
}
