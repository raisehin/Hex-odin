package main

// D3D12 Q8_0 GEMM microbench: the same weight tensor and the same quantised
// activations go through phonon's CPU kernels and through the D3D12 compute
// shader, so the only variable left is the arithmetic engine.
//
//   kern gpu <gguf> [tensor] [frames] [reps] [threads]
//
// Defaults to the first encoder block's FFN input projection, 103 frames (one
// second of audio after subsampling), 3 timed reps and the default pool size.

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:time"
import "../../phonon"
import "../../phonon/d3d12"

DEFAULT_TENSOR :: "enc.blocks.0.ff1.linear1.weight"

cmd_gpu :: proc(args: []string) {
	if len(args) < 3 { usage() }
	path := args[2]
	name := DEFAULT_TENSOR
	frames := 103
	reps := 3
	workers := 0
	if len(args) > 3 { name = args[3] }
	if len(args) > 4 { frames, _ = strconv.parse_int(args[4]) }
	if len(args) > 5 { reps, _ = strconv.parse_int(args[5]) }
	if len(args) > 6 { workers, _ = strconv.parse_int(args[6]) }

	g, err := phonon.gguf_load(path)
	if err != "" { fmt.eprintln(err); os.exit(1) }
	defer phonon.gguf_destroy(&g)

	w, ok := g.tensors[name]
	if !ok {
		fmt.eprintln("no tensor", name)
		os.exit(1)
	}
	if w.type != .Q8_0 {
		fmt.eprintln("tensor", name, "is", w.type, "- only Q8_0 is wired to the GPU GEMM")
		os.exit(1)
	}

	n := int(w.dims[1])
	k := int(w.dims[0])
	if k > d3d12.MAX_K {
		fmt.eprintln("k", k, "exceeds the shader's MAX_K", d3d12.MAX_K)
		os.exit(1)
	}

	pool := phonon.thread_pool_make(workers)
	defer phonon.thread_pool_free(pool)
	nthreads := pool == nil ? 1 : len(pool.threads)

	x := phonon.mat_make(frames, k)
	defer delete(x.data)
	// Deterministic, non-degenerate input; the exact distribution does not
	// matter, only that it exercises both signs and a wide dynamic range.
	seed: u32 = 0x1234_5678
	for i in 0 ..< len(x.data) {
		seed = seed * 1664525 + 1013904223
		x.data[i] = (f32(i32(seed >> 8) & 0xFFFF) / 32768.0 - 1.0) * 3.0
	}

	acts := phonon.act_make(frames, k)
	defer phonon.act_destroy(&acts)
	phonon.quantize_acts(acts, x)

	cpu_out := phonon.mat_make(frames, n)
	defer delete(cpu_out.data)

	// CPU reference and timing. This is the number the GPU has to beat.
	cpu_ms: f64 = 1e30
	for _ in 0 ..< reps {
		t0 := time.tick_now()
		phonon.lin_rows_par(cpu_out, x, w, nil, pool, nthreads)
		ms := time.duration_milliseconds(time.tick_since(t0))
		if ms < cpu_ms { cpu_ms = ms }
	}

	gpu_dev, gerr := d3d12.gpu_init(d3d12.Q8_0_GEMM_DXIL)
	if gerr != "" { fmt.eprintln("gpu_init:", gerr); os.exit(1) }
	defer d3d12.gpu_destroy(gpu_dev)

	gm, gmerr := d3d12.gemm_create(gpu_dev, n, k, frames, false)
	if gmerr != "" { fmt.eprintln("gemm_create:", gmerr); os.exit(1) }
	defer d3d12.gemm_destroy(gm)

	if werr := d3d12.gemm_set_weights(gm, w.bytes); werr != "" { fmt.eprintln("weights:", werr); os.exit(1) }
	if aerr := d3d12.gemm_set_activations(gm, acts.q, acts.d); aerr != "" { fmt.eprintln("acts:", aerr); os.exit(1) }

	gpu_out, run_err := d3d12.gemm_run(gm)
	if run_err != "" { fmt.eprintln("gemm_run:", run_err); os.exit(1) }

	gpu_ms: f64 = 1e30
	var_out: []f32
	for i in 0 ..< reps {
		if i > 0 { delete(gpu_out) }
		t0 := time.tick_now()
		gpu_out, run_err = d3d12.gemm_run(gm)
		if run_err != "" { fmt.eprintln("gemm_run:", run_err); os.exit(1) }
		ms := time.duration_milliseconds(time.tick_since(t0))
		if ms < gpu_ms { gpu_ms = ms }
	}
	var_out = gpu_out
	defer delete(var_out)

	// Numerics: both sides accumulate the same int32 block sums, so only the
	// f32 summation order differs.
	max_abs: f64
	sum_abs: f64
	max_rel: f64
	for i in 0 ..< len(var_out) {
		d := f64(var_out[i] - cpu_out.data[i])
		ad := math.abs(d)
		if ad > max_abs { max_abs = ad }
		sum_abs += ad
		den := math.abs(f64(cpu_out.data[i]))
		if den > 1e-3 {
			r := ad / den
			if r > max_rel { max_rel = r }
		}
	}
	mean_abs := sum_abs / f64(len(var_out))

	flops := 2.0 * f64(frames) * f64(n) * f64(k)
	fmt.printf("TENSOR %s ne0=%d ne1=%d frames=%d\n", name, k, n, frames)
	fmt.printf("KERN gpu_q8_0 tensor=%s frames=%d n=%d k=%d cpu_ms=%.2f cpu_gflops=%.2f gpu_ms=%.2f gpu_gflops=%.2f speedup=%.2f threads=%d reps=%d\n",
		name, frames, n, k,
		cpu_ms, flops / (cpu_ms * 1e6),
		gpu_ms, flops / (gpu_ms * 1e6),
		cpu_ms / gpu_ms, nthreads, reps)
	fmt.printf("NUMERICS gpu_vs_cpu max_abs=%.3e mean_abs=%.3e max_rel=%.3e\n", max_abs, mean_abs, max_rel)
}

// cmd_gpupeak measures the iGPU's dot4add_i8packed issue rate with no memory
// traffic in the loop, so the GEMM numbers can be read against the hardware
// ceiling instead of against the CPU.
cmd_gpupeak :: proc(args: []string) {
	groups := 4096
	iters := 1024
	reps := 3
	if len(args) > 2 { groups, _ = strconv.parse_int(args[2]) }
	if len(args) > 3 { iters, _ = strconv.parse_int(args[3]) }
	if len(args) > 4 { reps, _ = strconv.parse_int(args[4]) }

	g, err := d3d12.gpu_init(d3d12.Q8_0_GEMM_DXIL)
	if err != "" { fmt.eprintln("gpu_init:", err); os.exit(1) }
	defer d3d12.gpu_destroy(g)

	p, perr := d3d12.peak_create(g)
	if perr != "" { fmt.eprintln("peak_create:", perr); os.exit(1) }
	defer d3d12.peak_destroy(p)

	_ = d3d12.peak_run(p, u32(iters), u32(groups))
	best: f64 = 1e30
	for _ in 0 ..< reps {
		t0 := time.tick_now()
		if rerr := d3d12.peak_run(p, u32(iters), u32(groups)); rerr != "" {
			fmt.eprintln("peak_run:", rerr)
			os.exit(1)
		}
		ms := time.duration_milliseconds(time.tick_since(t0))
		if ms < best { best = ms }
	}

	threads := f64(groups) * f64(d3d12.PEAK_THREADS)
	dot4 := threads * f64(iters) * 8.0
	gflops := dot4 * 4.0 * 2.0 / (best * 1e6)
	fmt.printf("PEAK groups=%d threads=%.0f iters=%d ms=%.3f gdot4_per_s=%.2f gflops=%.1f\n",
		groups, threads, iters, best, dot4 / (best * 1e6), gflops)
}

// --- dictation check -------------------------------------------------------

gpuenc_cpu_final: []f32
gpuenc_gpu_final: []f32

gpuenc_trace_cpu :: proc(name: string, m: phonon.Mat) {
	if name != "enc.final" { return }
	delete(gpuenc_cpu_final)
	gpuenc_cpu_final = make([]f32, len(m.data))
	copy(gpuenc_cpu_final, m.data)
}

gpuenc_trace_gpu :: proc(name: string, m: phonon.Mat) {
	if name != "enc.final" { return }
	delete(gpuenc_gpu_final)
	gpuenc_gpu_final = make([]f32, len(m.data))
	copy(gpuenc_gpu_final, m.data)
}

// cmd_gpuenc runs one utterance through the CPU engine and through the GPU
// engine (HEX_GPU=1) and diffs the transcript and the encoder output. This
// is the dictation-inference gate: same words, bounded numeric drift, and the
// encoder milliseconds for both paths.
cmd_gpuenc :: proc(args: []string) {
	if len(args) < 4 { usage() }
	workers := 0
	if len(args) > 4 { workers, _ = strconv.parse_int(args[4]) }

	raw, _, ok := wav_read_mono16k(args[3])
	if !ok { fmt.eprintln("cannot read", args[3]); os.exit(1) }
	samples, trimmed_ms := trim_trailing_silence(raw)

	os.set_env("HEX_GPU", "0")
	ec, cerr := phonon.engine_open(args[2], workers)
	if cerr != "" { fmt.eprintln("engine cpu:", cerr); os.exit(1) }
	phonon.set_trace(&ec, gpuenc_trace_cpu)
	res_cpu := phonon.transcribe(&ec, samples)
	cpu_enc_ms := res_cpu.enc_ms
	phonon.engine_close(&ec)

	os.set_env("HEX_GPU", "1")
	eg, gerr := phonon.engine_open(args[2], workers)
	if gerr != "" { fmt.eprintln("engine gpu:", gerr); os.exit(1) }
	defer phonon.engine_close(&eg)
	gpu_on, gpu_open_err := phonon.engine_gpu_status(&eg)
	if !gpu_on {
		fmt.eprintln("gpu engine fell back to CPU at open:", gpu_open_err)
		os.exit(1)
	}
	phonon.set_trace(&eg, gpuenc_trace_gpu)
	// The first GPU utterance uploads every weight; time the steady state.
	_ = phonon.transcribe(&eg, samples)
	stats: phonon.Encode_Stats
	phonon.set_stats(&eg, &stats)
	t1 := time.tick_now()
	res_gpu := phonon.transcribe(&eg, samples)
	gpu_enc_ms := ms_since(t1) - res_gpu.mel_ms - res_gpu.dec_ms

	match := res_cpu.text == res_gpu.text
	max_abs: f64
	sum_abs: f64
	max_rel: f64
	if len(gpuenc_cpu_final) == len(gpuenc_gpu_final) && len(gpuenc_cpu_final) > 0 {
		for i in 0 ..< len(gpuenc_gpu_final) {
			d := f64(gpuenc_gpu_final[i] - gpuenc_cpu_final[i])
			ad := math.abs(d)
			if ad > max_abs { max_abs = ad }
			sum_abs += ad
			den := math.abs(f64(gpuenc_cpu_final[i]))
			if den > 1e-3 {
				r := ad / den
				if r > max_rel { max_rel = r }
			}
		}
	}
	mean_abs := sum_abs / f64(max(len(gpuenc_gpu_final), 1))
	fmt.printf("GPUENC transcript_match=%v cpu_tokens=%d gpu_tokens=%d trimmed_ms=%d\n",
		match, len(res_cpu.tokens), len(res_gpu.tokens), trimmed_ms)
	fmt.printf("GPUENC cpu_text=%q\ngpu_text=%q\n", res_cpu.text, res_gpu.text)
	fmt.printf("GPUENC cpu_enc_ms=%.1f gpu_enc_ms=%.1f speedup=%.2f\n",
		cpu_enc_ms, gpu_enc_ms, cpu_enc_ms / gpu_enc_ms)
	fmt.printf("GPUENC enc_final max_abs=%.3e mean_abs=%.3e max_rel=%.3e\n", max_abs, mean_abs, max_rel)
	fmt.printf("GPUENC stages attn_qkv=%.1f attn_score=%.1f ffn=%.1f conv=%.1f norm=%.1f resid=%.1f\n",
		stats.attn_qkv, stats.attn_score, stats.ffn, stats.conv, stats.norm, stats.resid)
	fmt.printf("GPUENC attn_split sc=%.1f av=%.1f out=%.1f\n",
		stats.attn_sc, stats.attn_av, stats.attn_out)
	split_n, split_up, split_launch, split_span, split_join := phonon.gpu_split_stats(eg.enc.gpu)
	split_total := split_up + split_launch + split_span + split_join
	fmt.printf("GPUENC split n=%d up=%.1f launch=%.1f span=%.1f join=%.1f sum=%.1f (enc %.1f)\n",
		split_n, split_up, split_launch, split_span, split_join, split_total, gpu_enc_ms)
	if split_n > 0 {
		fmt.printf("GPUENC split_per_linear up=%.3f launch=%.3f span=%.3f join=%.3f ms\n",
			split_up / f64(split_n), split_launch / f64(split_n),
			split_span / f64(split_n), split_join / f64(split_n))
	}
}
