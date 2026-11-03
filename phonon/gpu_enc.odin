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
// Selection is explicit: HEX_GPU=1 (or d3d12/gpu) enables the device at engine
// open. Unset means Center CPU, no device, no extra memory.

package phonon

import "core:fmt"
import "core:mem"
import "core:os"
import "core:time"
import d3d12 "./d3d12"

// GPU_ROW_PCT is the percentage of output rows handed to the GPU. It is not
// 50: measuring the split internals (stats.gpu_span vs stats.gpu_join) shows
// the CPU half finishes about 30% sooner than the GPU per row, so an even
// split leaves the GPU as the straggler and the join waits on it. Giving the
// GPU fewer rows balances the two halves. Must stay even-aligned.
GPU_ROW_PCT :: 50

// split_mid returns the GPU-owned row count for n output rows, even so the
// CPU half starts on a pair boundary.
split_mid :: proc(n: int) -> int {
	return (n * GPU_ROW_PCT / 100) / 2 * 2
}

// gms is the local millisecond helper (encoder.odin's ms_since is file-private).
@(private = "file")
gms :: proc(t: time.Tick) -> f64 {
	return time.duration_milliseconds(time.tick_since(t))
}

Gpu_Enc :: struct {
	dev:     ^d3d12.Gpu,
	ok:      bool,
	err:     string,
	weights: map[string]^d3d12.Weights,
	scratch: map[[3]int]^d3d12.Scratch,
}

gpu_enc_wanted :: proc() -> bool {
	v := os.get_env("HEX_GPU", context.temp_allocator)
	return v == "1" || v == "d3d12" || v == "gpu"
}

gpu_enc_open :: proc() -> ^Gpu_Enc {
	ge := new(Gpu_Enc)
	ge.weights = make(map[string]^d3d12.Weights)
	ge.scratch = make(map[[3]int]^d3d12.Scratch)
	dev, err := d3d12.gpu_init(d3d12.Q8_0_GEMM_DXIL)
	if err != "" {
		ge.err = err
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

gpu_enc_close :: proc(ge: ^Gpu_Enc) {
	if ge == nil { return }
	for _, s in ge.scratch { d3d12.scratch_destroy(s) }
	delete(ge.scratch)
	for _, w in ge.weights { d3d12.weights_destroy(w) }
	delete(ge.weights)
	d3d12.gpu_destroy(ge.dev)
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
gpu_lin_rows :: proc(out, x: Mat, w: Gguf_Tensor, bias: []f32, ge: ^Gpu_Enc, pool: ^Pool, workers: int, st: ^Encode_Stats = nil) -> bool {
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
	mid := split_mid(n)
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
	_tup := time.tick_now()
	if err := d3d12.scratch_write_acts(sc, mem.slice_data_cast([]u8, acts.q), mem.slice_data_cast([]u8, acts.d)); err != "" {
		ge.err = err
		ge.ok = false
		return false
	}
	if st != nil { st.gpu_up += gms(_tup) }
	if err := d3d12.gemm_launch(ge.dev, ww, sc, mid); err != "" {
		// Launch failed before the GPU did anything: whole linear on CPU.
		ge.err = fmt.tprintf("gpu exec %s: %s", w.name, err)
		ge.ok = false
		lin_rows_par(out, x, w, bias, pool, workers)
		return true
	}
	// The pool owns rows [mid, n) while the GPU owns [0, mid).
	_ts := time.tick_now()
	lin_rows_par_span(out, x, w, bias, acts, pool, workers, mid, n)
	if st != nil { st.gpu_span += gms(_ts) }
	_tj := time.tick_now()
	if err := d3d12.gemm_finish_strided(ge.dev, sc, out.data, n); err != "" {
		// GPU failed after the CPU half finished: fill its rows on the CPU.
		ge.err = fmt.tprintf("gpu exec %s: %s", w.name, err)
		ge.ok = false
		lin_rows_par_span(out, x, w, bias, acts, pool, workers, 0, mid)
		return true
	}
	if st != nil { st.gpu_join += gms(_tj) }
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

	// Even split point, same gate and ratio as gpu_lin_rows: the CPU half of
	// every projection must stay bit-identical with the serial path.
	mid := split_mid(n)
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
	if err := d3d12.scratch_ensure_slots(sc, 3); err != "" {
		ge.err = err
		ge.ok = false
		return false
	}

	// One quantisation of the shared input, one upload; all three GEMMs read
	// the same act_q/act_d pair.
	acts := act_make(t, kk, context.temp_allocator)
	defer act_destroy(&acts, context.temp_allocator)
	quantize_acts_par(acts, x, pool, workers)
	_tup := time.tick_now()
	if err := d3d12.scratch_write_acts(sc, mem.slice_data_cast([]u8, acts.q), mem.slice_data_cast([]u8, acts.d)); err != "" {
		ge.err = err
		ge.ok = false
		for i in 0 ..< 3 { lin_rows_par(outs[i], x, ws[i], nil, pool, workers) }
		return true
	}
	if e.stats != nil { e.stats.gpu_up += gms(_tup) }

	// One submission, one fence: all three projections are recorded into a
	// single command list and land in separate output slots. The three CPU
	// halves then run back-to-back and a single join covers all of them,
	// instead of launch/span/join three times.
	if err := d3d12.gemm_launch_batch3(ge.dev, wws, sc, mid); err != "" {
		// Nothing reached the GPU: whole trio on the CPU.
		ge.err = fmt.tprintf("gpu qkv batch: %s", err)
		ge.ok = false
		for i in 0 ..< 3 { lin_rows_par(outs[i], x, ws[i], nil, pool, workers) }
		return true
	}
	_ts := time.tick_now()
	for i in 0 ..< 3 {
		// The pool owns rows [mid, n) while the GPU owns [0, mid).
		lin_rows_par_span(outs[i], x, ws[i], nil, acts, pool, workers, mid, n)
	}
	if e.stats != nil { e.stats.gpu_span += gms(_ts) }

	_tj := time.tick_now()
	for i in 0 ..< 3 {
		if err := d3d12.gemm_finish_strided_slot(ge.dev, sc, outs[i].data, n, i); err != "" {
			ge.err = fmt.tprintf("gpu exec %s: %s", ws[i].name, err)
			ge.ok = false
			lin_rows_par_span(outs[i], x, ws[i], nil, acts, pool, workers, 0, mid)
			for j in i + 1 ..< 3 { lin_rows_par(outs[j], x, ws[j], nil, pool, workers) }
			return true
		}
	}
	if e.stats != nil { e.stats.gpu_join += gms(_tj) }
	return true
}

// enc_lin_rows is the encoder's linear: GPU+CPU row-split when the sidecar
// is live and the shape qualifies, the pooled CPU kernels otherwise.
enc_lin_rows :: proc(e: ^Encoder, out, x: Mat, w: Gguf_Tensor, bias: []f32) {
	if gpu_lin_rows(out, x, w, bias, e.gpu, e.pool, e.workers, e.stats) { return }
	lin_rows_par(out, x, w, bias, e.pool, e.workers)
}

// engine_gpu_status reports whether dictation inference is using the GPU.
// A nil sidecar (HEX_GPU unset) is not an error: it is the CPU default.
engine_gpu_status :: proc(e: ^Engine) -> (enabled: bool, err: string) {
	if e.enc.gpu == nil { return false, "" }
	return e.enc.gpu.ok, e.enc.gpu.err
}
