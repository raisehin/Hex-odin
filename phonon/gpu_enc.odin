// GPU sidecar for the encoder: persistent Q8_0 weights, per-shape scratch.
//
// The D3D12 layer owns buffers; this file owns policy. Weights are uploaded
// once per tensor and read by every utterance until the engine closes, so the
// 600 MB repack is a load-time cost, not a per-utterance one. Scratch buffers
// are keyed by (n, k, t) and reused whenever an utterance has the same
// geometry. Anything the GPU cannot do — a non-Q8_0 tensor, an oversized k,
// an upload failure, a dispatch error — returns false and the caller runs the
// CPU kernels instead; a dispatch error additionally poisons the sidecar so
// the rest of the utterance stays on the CPU path.
//
// The device is opened by default. HEX_GPU=0 (or off/false/no/cpu/none) opts out
// and keeps the engine CPU-only; any value that is not an explicit opt-out tries
// the device, and a failed open leaves a nil sidecar, which every caller already
// treats as "run the CPU kernel".

package phonon

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"
import "core:time"
import d3d12 "./d3d12"

Gpu_Enc :: struct {
	dev:     ^d3d12.Gpu,
	ok:      bool,
	err:     string,
	// Measurement only: what a split linear actually spends its time on.
	// up is the activation upload, launch is the dispatch, span is the CPU
	// half running concurrently with the GPU, join is the fence wait plus
	// the readback of the GPU's rows. If join is a large fraction of the
	// total, the cost is the per-linear round trip, not the arithmetic.
	n_engaged: int,
	up_ms:     f64,
	launch_ms: f64,
	span_ms:   f64,
	join_ms:   f64,
	weights: map[string]^d3d12.Weights,
	scratch: map[[3]int]^d3d12.Scratch,
}

// gpu_enc_wanted reports whether to try opening the D3D12 sidecar. The hybrid
// split is ON by default: a device that cannot be opened, or a failure part-way
// through an utterance, poisons the sidecar and the rest of that run falls back
// to the CPU kernels, so a machine without a usable GPU pays one failed open
// and then behaves exactly as before. HEX_GPU=0 (or off/false/no/cpu/none) opts
// out and keeps the engine CPU-only -- which is what the benchmark uses to pin
// its CPU arm.
gpu_enc_wanted :: proc() -> bool {
	v := os.get_env("HEX_GPU", context.temp_allocator)
	switch v {
	case "0", "off", "false", "no", "cpu", "none":
		return false
	}
	return true
}

gpu_enc_open :: proc() -> ^Gpu_Enc {
	ge := new(Gpu_Enc)
	ge.weights = make(map[string]^d3d12.Weights)
	ge.scratch = make(map[[3]int]^d3d12.Scratch)
	dev, err := d3d12.gpu_init(d3d12.Q8_0_GEMM_DXIL)
	if err != "" {
		// gpu_init reports through fmt.tprintf (the temporary allocator), so
		// the message must be copied before this scope ends.
		ge.err = strings.clone(err)
		return ge
	}
	ge.dev = dev
	ge.ok = true
	return ge
}

gpu_enc_open_if_enabled :: proc() -> ^Gpu_Enc {
	if !gpu_enc_wanted() { return nil }
	return gpu_enc_open()
}

// gpu_split_stats reports how many linears took the split and what each phase
// cost, so the benchmark can separate arithmetic from per-linear round trips.
gpu_split_stats :: proc(ge: ^Gpu_Enc) -> (n: int, up, launch, span, join: f64) {
	if ge == nil { return }
	return ge.n_engaged, ge.up_ms, ge.launch_ms, ge.span_ms, ge.join_ms
}

gpu_enc_close :: proc(ge: ^Gpu_Enc) {
	if ge == nil { return }
	for _, s in ge.scratch { d3d12.scratch_destroy(s) }
	delete(ge.scratch)
	for _, w in ge.weights { d3d12.weights_destroy(w) }
	delete(ge.weights)
	d3d12.gpu_destroy(ge.dev)
	if ge.err != "" { delete(ge.err) }
	free(ge)
}

// SPLIT_MIN_MAC floors the row-split in multiply-accumulates (t*n*k): the
// split halves compute but pays one launch, one fence and one readback, so
// small linears stay on the CPU path. Measured on the 680M: below ~80M MACs
// the overhead exceeds the concurrency gain (short commands run pure CPU
// with zero regression); above it the split wins.
SPLIT_MIN_MAC :: 80_000_000

// gpu_lin_rows runs one encoder linear as a GPU+CPU row-split: the GPU takes
// output rows [0, mid) while the pool takes [mid, n) concurrently, sharing a
// single activation quantisation. True means out is filled; false means the
// caller must run the CPU kernels. A GPU failure mid-flight fills the missing
// rows on the CPU and poisons the sidecar, so the rest of the utterance stays
// on the CPU path.
gpu_lin_rows :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32, ge: ^Gpu_Enc, pool: ^Pool, workers: int) -> bool {
	if ge == nil || !ge.ok { return false }
	if w.type != .Q8_0 { return false }
	n := int(w.dims[1])
	k := int(w.dims[0])
	t := x.rows
	if n <= 0 || k <= 0 || t <= 0 { return false }
	if k % 32 != 0 || k > d3d12.MAX_K { return false }
	if out.rows != t || out.cols != n || x.cols != k { return false }
	if len(out.data) != t * n { return false }
	has_bias := bias != nil
	if has_bias && len(bias) != n { return false }
	if len(w.bytes) != n * (k / 32) * 34 { return false }

	ww, found := ge.weights[w.name]
	if !found {
		nw, err := d3d12.weights_upload(ge.dev, n, k, w.bytes, has_bias)
		if err != "" {
			ge.err = fmt.tprintf("gpu weights %s: %s", w.name, err)
			ge.ok = false
			return false
		}
		if has_bias {
			if berr := d3d12.weights_set_bias(nw, bias); berr != "" {
				d3d12.weights_destroy(nw)
				ge.err = berr
				ge.ok = false
				return false
			}
		}
		ge.weights[w.name] = nw
		ww = nw
	}
	if ww.n != n || ww.k != k || ww.has_bias != has_bias { return false }

	// Even split point: lin_rows_core pairs rows, so every range must start
	// on an even row to stay bit-identical with the serial path.
	mid := (n / 2) / 2 * 2
	if mid < 2 || t*n*k < SPLIT_MIN_MAC { return false }

	key := [3]int{mid, k, t}
	sc, sok := ge.scratch[key]
	if !sok {
		ns, err := d3d12.scratch_create(ge.dev, mid, k, t)
		if err != "" {
			ge.err = err
			ge.ok = false
			return false
		}
		ge.scratch[key] = ns
		sc = ns
	}

	acts := act_make(t, k, context.temp_allocator)
	defer act_destroy(&acts, context.temp_allocator)
	quantize_acts_par(acts, x, pool, workers)
	_t_up := time.tick_now()
	if err := d3d12.scratch_write_acts(sc, mem.slice_data_cast([]u8, acts.q), mem.slice_data_cast([]u8, acts.d)); err != "" {
		ge.err = err
		ge.ok = false
		return false
	}
	ge.up_ms += time.duration_milliseconds(time.tick_since(_t_up))
	_t_launch := time.tick_now()
	if err := d3d12.gemm_launch(ge.dev, ww, sc, mid); err != "" {
		// Launch failed before the GPU did anything: whole linear on CPU.
		ge.err = fmt.tprintf("gpu exec %s: %s", w.name, err)
		ge.ok = false
		lin_rows_par(out, x, w, bias, pool, workers)
		return true
	}
	ge.launch_ms += time.duration_milliseconds(time.tick_since(_t_launch))
	// The pool owns rows [mid, n) while the GPU owns [0, mid).
	_t_span := time.tick_now()
	lin_rows_par_span(out, x, w, bias, acts, pool, workers, mid, n)
	ge.span_ms += time.duration_milliseconds(time.tick_since(_t_span))
	_t_join := time.tick_now()
	if err := d3d12.gemm_finish_strided(ge.dev, sc, out.data, n); err != "" {
		// GPU failed after the CPU half finished: fill its rows on the CPU.
		ge.err = fmt.tprintf("gpu exec %s: %s", w.name, err)
		ge.ok = false
		lin_rows_par_span(out, x, w, bias, acts, pool, workers, 0, mid)
		return true
	}
	ge.join_ms += time.duration_milliseconds(time.tick_since(_t_join))
	ge.n_engaged += 1
	return true
}

// gpu_qkv_rows runs the three attention projections, which share the same
// input x, as one quantisation + one activation upload followed by three
// sequential row-splits over the shared scratch. Saves 2x quantize_acts +
// 2x scratch_write_acts per block vs three gpu_lin_rows calls; the launches
// and joins stay sequential because the three outputs share one out/readback
// pair (each finish consumes it before the next launch overwrites it).
// True means all three outs are filled (GPU split or CPU fallback inside);
// false means nothing was done and the caller must run the CPU kernels.
gpu_qkv_rows :: proc(e: ^Encoder, q, k, v, x: Mat, wq, wk, wv: Gguf_Tensor) -> bool {
	ge := e.gpu
	pool := e.pool
	workers := e.workers
	if ge == nil || !ge.ok { return false }
	if wq.type != .Q8_0 || wk.type != .Q8_0 || wv.type != .Q8_0 { return false }
	n := int(wq.dims[1])
	kk := int(wq.dims[0])
	t := x.rows
	if n <= 0 || kk <= 0 || t <= 0 { return false }
	if int(wk.dims[1]) != n || int(wv.dims[1]) != n { return false }
	if int(wk.dims[0]) != kk || int(wv.dims[0]) != kk { return false }
	if kk % 32 != 0 || kk > d3d12.MAX_K { return false }
	if x.cols != kk { return false }
	outs := [3]Mat{q, k, v}
	ws := [3]Gguf_Tensor{wq, wk, wv}
	for i in 0 ..< 3 {
		if outs[i].rows != t || outs[i].cols != n { return false }
		if len(outs[i].data) != t * n { return false }
		if len(ws[i].bytes) != n * (kk / 32) * 34 { return false }
	}

	// Even split point, same gate as gpu_lin_rows: the CPU half of every
	// projection must stay bit-identical with the serial path.
	mid := (n / 2) / 2 * 2
	if mid < 2 || t*n*kk < SPLIT_MIN_MAC { return false }

	wws := [3]^d3d12.Weights{}
	for i in 0 ..< 3 {
		ww, found := ge.weights[ws[i].name]
		if !found {
			nw, err := d3d12.weights_upload(ge.dev, n, kk, ws[i].bytes, false)
			if err != "" {
				ge.err = fmt.tprintf("gpu weights %s: %s", ws[i].name, err)
				ge.ok = false
				return false
			}
			ge.weights[ws[i].name] = nw
			ww = nw
		}
		if ww.n != n || ww.k != kk || ww.has_bias { return false }
		wws[i] = ww
	}

	key := [3]int{mid, kk, t}
	sc, sok := ge.scratch[key]
	if !sok {
		ns, err := d3d12.scratch_create(ge.dev, mid, kk, t)
		if err != "" {
			ge.err = err
			ge.ok = false
			return false
		}
		ge.scratch[key] = ns
		sc = ns
	}

	// One quantisation of the shared input, one upload; all three GEMMs read
	// the same act_q/act_d pair.
	acts := act_make(t, kk, context.temp_allocator)
	defer act_destroy(&acts, context.temp_allocator)
	quantize_acts_par(acts, x, pool, workers)
	if err := d3d12.scratch_write_acts(sc, mem.slice_data_cast([]u8, acts.q), mem.slice_data_cast([]u8, acts.d)); err != "" {
		ge.err = err
		ge.ok = false
		for i in 0 ..< 3 { lin_rows_par(outs[i], x, ws[i], nil, pool, workers) }
		return true
	}
	for i in 0 ..< 3 {
		if err := d3d12.gemm_launch(ge.dev, wws[i], sc, mid); err != "" {
			// Launch failed before the GPU did anything: this linear and the
			// rest run on the CPU. Earlier outs are already filled.
			ge.err = fmt.tprintf("gpu exec %s: %s", ws[i].name, err)
			ge.ok = false
			for j in i ..< 3 { lin_rows_par(outs[j], x, ws[j], nil, pool, workers) }
			return true
		}
		// The pool owns rows [mid, n) while the GPU owns [0, mid).
		lin_rows_par_span(outs[i], x, ws[i], nil, acts, pool, workers, mid, n)
		if err := d3d12.gemm_finish_strided(ge.dev, sc, outs[i].data, n); err != "" {
			ge.err = fmt.tprintf("gpu exec %s: %s", ws[i].name, err)
			ge.ok = false
			lin_rows_par_span(outs[i], x, ws[i], nil, acts, pool, workers, 0, mid)
			for j in i + 1 ..< 3 { lin_rows_par(outs[j], x, ws[j], nil, pool, workers) }
			return true
		}
	}
	return true
}

// enc_lin_rows is the encoder's linear: GPU+CPU row-split when the sidecar
// is live and the shape qualifies, the pooled CPU kernels otherwise.
enc_lin_rows :: proc(e: ^Encoder, out, x: Mat, w: Gguf_Tensor, bias: []f32) {
	if gpu_lin_rows(out, x, w, bias, e.gpu, e.pool, e.workers) { return }
	lin_rows_par(out, x, w, bias, e.pool, e.workers)
}

// engine_gpu_status reports whether dictation inference is using the GPU.
// A nil sidecar (HEX_GPU unset) is not an error: it is the CPU default.
engine_gpu_status :: proc(e: ^Engine) -> (enabled: bool, err: string) {
	if e.enc.gpu == nil { return false, "" }
	return e.enc.gpu.ok, e.enc.gpu.err
}
