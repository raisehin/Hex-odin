package main

// Kernel/end-to-end bench driver for the phonon port.
//   kern encoder <gguf> [frames] [threads]   - time every encoder matmul path
//   kern transcribe <gguf> <wav> [threads]   - full engine, PROJECT line on stderr
//   kern parity <gguf> <wav> <dump_dir>      - stage diff vs transcribe.cpp dumps
//
// Every number this prints must come from the phonon package itself, not from a
// standalone copy of the kernel, so a change in phonon/ shows up here.

import "core:fmt"
import "core:math"
import "core:mem"
import "core:os"
import "core:strconv"
import "core:strings"
import "core:time"
import "../../phonon"

ms_since :: proc(t: time.Tick) -> f64 {
	return time.duration_milliseconds(time.tick_since(t))
}

usage :: proc() {
	fmt.eprintln("usage: kern encoder <gguf> [frames] [threads]")
	fmt.eprintln("       kern transcribe <gguf> <wav> [threads]")
	fmt.eprintln("       kern parity <gguf> <wav> <dump_dir>")
	fmt.eprintln("       kern parity-enc <gguf> <dump_dir>   (encoder stages only, ref mel in)")
	fmt.eprintln("       kern parcheck <gguf> [threads] [tensor]")
	fmt.eprintln("       kern gpu <gguf> [tensor] [frames] [reps] [threads]")
	fmt.eprintln("       kern gpupeak [groups] [iters] [reps]")
	fmt.eprintln("       kern gpuenc <gguf> <wav> [threads]  (CPU vs HEX_GPU=1 dictation check)")
	fmt.eprintln("       kern enchash <gguf> [frames] [threads]  (deterministic encode, FNV hash oracle)")
	os.exit(2)
}

main :: proc() {
	args := os.args
	if len(args) < 3 { usage() }
	switch args[1] {
	case "encoder":    cmd_encoder(args)
	case "transcribe": cmd_transcribe(args)
	case "parity":     cmd_parity(args)
	case "parity-enc": cmd_parity_enc(args)
	case "types":      cmd_types(args)
	case "parcheck":   cmd_parcheck(args)
	case "gpu":        cmd_gpu(args)
	case "gpupeak":    cmd_gpupeak(args)
	case "gpuenc":     cmd_gpuenc(args)
	case "enchash":    cmd_enchash(args)
	case:              usage()
	}
}

// --- bit-identity oracle ---------------------------------------------------

// cmd_enchash runs the encoder over a deterministic synthetic mel plane and
// prints an FNV-1a hash of the output bits. Two builds that print the same
// hash computed bit-identical results; any summation-order change shows up as
// a different hash even when the transcript survives. This is the check the
// "bit-identical by construction" claims are verified against.
cmd_enchash :: proc(args: []string) {
	model := args[2]
	frames := 132
	workers := 8
	if len(args) > 3 { frames, _ = strconv.parse_int(args[3]) }
	if len(args) > 4 { workers, _ = strconv.parse_int(args[4]) }

	e, err := phonon.engine_open(model, workers)
	if err != "" { fmt.eprintln("engine:", err); os.exit(1) }
	defer phonon.engine_close(&e)

	n_mels := 128
	mel := make([]f32, n_mels * frames, context.temp_allocator)
	// Deterministic LCG so the input is identical across builds and runs.
	s := u32(0x1234_5678)
	for &v in mel {
		s = s * 1664525 + 1013904223
		v = f32(i32(s >> 8) - 0x80_0000) / f32(0x80_0000)
	}

	enc := phonon.encode(&e.enc, mel, n_mels, frames, context.temp_allocator)
	fmt.printf("ENCHASH frames=%d threads=%d rows=%d cols=%d\n", frames, workers, enc.rows, enc.cols)

	h := u32(2166136261)
	nan := 0
	for v in enc.data {
		bits := transmute(u32)v
		// non-finite: exponent all ones
		if (bits & 0x7F80_0000) == 0x7F80_0000 { nan += 1 }
		h = (h ~ bits) * 16777619
	}
	fmt.printf("ENCHASH hash=%08x elements=%d nonfinite=%d\n", h, len(enc.data), nan)
}

// --- encoder matmul bench --------------------------------------------------

// cmd_encoder loads the GGUF and pushes every 2-D encoder matmul through
// phonon.lin_rows_par, reporting ms per weight type and in total. This is the
// same entry point the encoder uses, so kernel and scheduling changes both land
// in the number.
cmd_encoder :: proc(args: []string) {
	frames := 13
	workers := 0
	passes_n := 5
	if len(args) > 3 { frames, _ = strconv.parse_int(args[3]) }
	if len(args) > 4 { workers, _ = strconv.parse_int(args[4]) }
	if len(args) > 5 { passes_n, _ = strconv.parse_int(args[5]) }

	g, err := phonon.gguf_load(args[2])
	if err != "" { fmt.eprintln(err); os.exit(1) }
	defer phonon.gguf_destroy(&g)
	pool := phonon.thread_pool_make(workers)
	defer phonon.thread_pool_free(pool)
	nthreads := pool == nil ? 1 : len(pool.threads)

	total: f64
	macs: f64
	per_type: map[phonon.GGML_Type]f64
	count_type: map[phonon.GGML_Type]int
	bytes_type: map[phonon.GGML_Type]int
	reps :: 3
	// One pass over every tensor is a noisy sample on a shared laptop: repeat
	// the whole sweep and report the spread, so a change smaller than the
	// run-to-run drift is still visible as a min shift.
	pass_totals := make([dynamic]f64)
	defer delete(pass_totals)
	for pass in 0 ..< passes_n {
	pass_sum: f64 = 0
	for name in g.order {
		t := g.tensors[name]
		if t.n_dims != 2 || (t.type != .Q8_0 && t.type != .F16 && t.type != .F32) { continue }
		if len(name) < 4 || !strings.has_prefix(name, "enc.") { continue }
		cols := int(t.dims[0])
		rows := int(t.dims[1])
		x := phonon.mat_make(frames, cols)
		out := phonon.mat_make(frames, rows)
		defer phonon.mat_destroy(&x)
		defer phonon.mat_destroy(&out)
		for i in 0 ..< len(x.data) { x.data[i] = 0.01 * f32(i % 13) }

		phonon.lin_rows_par(out, x, t, nil, pool, nthreads) // warm
		start := time.tick_now()
		for _ in 0 ..< reps { phonon.lin_rows_par(out, x, t, nil, pool, nthreads) }
		el := ms_since(start) / f64(reps)
		if true {
		}
		total += el
		pass_sum += el
		macs += f64(rows) * f64(cols) * f64(frames)
			if len(name) > 22 && name[:22] == "enc.pre_encode.out.wei" { s: f64 = 0; for v in out.data { s += f64(v) }; fmt.printf("  chk %s threads=%d y0=%.6f sum=%.4f\n", name, nthreads, out.data[0], s) }
		per_type[t.type] += el
		count_type[t.type] += 1
		bytes_type[t.type] += len(t.bytes)
	}
	fmt.printf("frames %d  threads %d\n", frames, nthreads)
	types := [?]phonon.GGML_Type{.Q8_0, .F16, .F32}
	for t in types {
		if count_type[t] == 0 { continue }
		fmt.printf("  %-4v %3d tensors %6.1f MB %9.2f ms\n", t, count_type[t], f64(bytes_type[t])/1e6, per_type[t])
	}
	append(&pass_totals, pass_sum)
	}
	best := pass_totals[0]
	for v in pass_totals { if v < best { best = v } }
	mean: f64
	for v in pass_totals { mean += v }
	mean /= f64(len(pass_totals))
	if passes_n > 1 {
		fmt.printf("  passes:", )
		for v in pass_totals { fmt.printf(" %.1f", v) }
		fmt.println()
	}
	fmt.printf("KERN encoder_matmul frames=%d threads=%d total_ms=%.2f min_ms=%.2f mean_ms=%.2f passes=%d gflop=%.1f gflops=%.1f\n",
		frames, nthreads, best, best, mean, passes_n, 2 * macs / 1e9, 2 * macs / 1e9 / (best / 1000))
}


// trim_trailing_silence mirrors src/transcribe.odin so the phonon engine sees
// exactly the samples the shim sees (same clip, same endpoint policy).
TRIM_TAIL_PAD_MS :: 250
TRIM_MIN_CUT_MS :: 100
TRIM_FRAME_MS :: 20
TRIM_REL_FLOOR :: 0.001
TRIM_RUN_FRAMES :: 5

trim_trailing_silence :: proc(samples: []f32) -> (out: []f32, trimmed_ms: int) {
	out = samples
	rate := 16_000
	frame := rate * TRIM_FRAME_MS / 1000
	if len(samples) < frame * 8 { return }
	peak: f32 = 0
	frames := (len(samples) + frame - 1) / frame
	energy := make([]f32, frames, context.temp_allocator)
	for i in 0 ..< frames {
		off := i * frame
		acc: f32 = 0
		for s in samples[off:min(off + frame, len(samples))] { acc += s * s }
		energy[i] = acc / f32(frame)
		if energy[i] > peak { peak = energy[i] }
	}
	if peak < 1e-8 { return }
	limit := peak * f32(TRIM_REL_FLOOR)
	last := 0
	run := 0
	for i in 0 ..< frames {
		if energy[i] > limit {
			if run += 1; run >= TRIM_RUN_FRAMES { last = (i + 1) * frame }
		} else {
			run = 0
		}
	}
	keep := min(last + rate * TRIM_TAIL_PAD_MS / 1000, len(samples))
	cut := len(samples) - keep
	if cut < rate * TRIM_MIN_CUT_MS / 1000 { return }
	trimmed_ms = cut * 1000 / rate
	return samples[:keep], trimmed_ms
}

// --- end-to-end ------------------------------------------------------------

cmd_transcribe :: proc(args: []string) {
	if len(args) < 4 { usage() }
	workers := 0
	if len(args) > 4 { workers, _ = strconv.parse_int(args[4]) }

	raw, _, ok := wav_read_mono16k(args[3])
	if !ok { fmt.eprintln("cannot read", args[3]); os.exit(1) }
	samples, trimmed_ms := trim_trailing_silence(raw)

	t0 := time.tick_now()
	e, err := phonon.engine_open(args[2], workers)
	if err != "" { fmt.eprintln("engine:", err); os.exit(1) }
	defer phonon.engine_close(&e)
	model_ms := ms_since(t0)

	want_stats := len(args) > 5 && args[5] == "stats"
	stats: phonon.Encode_Stats
	if want_stats { phonon.set_stats(&e, &stats) }

	t1 := time.tick_now()
	res := phonon.transcribe(&e, samples)
	infer_ms := ms_since(t1)

	fmt.printf(
		"PROFILE model_ms=%d infer_ms=%d samples=%d engines=1\n",
		int(model_ms + 0.5), int(infer_ms + 0.5), len(samples),
	)
	fmt.eprintf(
		"PHONON_STAGES mel=%.1f enc=%.1f dec=%.1f tokens=%d T=%d raw=%d kept=%d trimmed_ms=%d\n",
		res.mel_ms, res.enc_ms, res.dec_ms, len(res.tokens), e.enc.T, len(raw), len(samples), trimmed_ms,
	)
	if want_stats {
		// attn_score spans attention() + linear_out, so attn_qkv is a subset of it.
		sum := stats.pre_encode + stats.norm + stats.ffn + stats.attn_score + stats.conv +
			stats.conv_glu + stats.conv_dw + stats.conv_pw2 + stats.resid
		fmt.printf(
			"ENC_STAGE calls=%d enc=%.1f total=%.1f sum=%.1f | pre=%.1f norm=%.1f ffn=%.1f (silu %.1f) attn=%.1f (qkv %.1f rest %.1f) conv=%.1f (pw1 %.1f glu %.1f dw %.1f pw2 %.1f) resid=%.1f\n",
			stats.calls, res.enc_ms, stats.total, sum, stats.pre_encode, stats.norm, stats.ffn, stats.silu,
			stats.attn_score, stats.attn_qkv, stats.attn_score - stats.attn_qkv, stats.conv +
			stats.conv_glu + stats.conv_dw + stats.conv_pw2, stats.conv, stats.conv_glu,
			stats.conv_dw, stats.conv_pw2, stats.resid,
		)
		c := stats.pre_conv[:]
		fmt.printf("PRE_STAGE calls=%d scatter=%.1f conv0=%.1f conv2=%.1f conv3=%.1f conv5=%.1f conv6=%.1f stack=%.1f out=%.1f\n",
			stats.calls, c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7])
	}
	fmt.println(res.text)
}

// --- parallel dispatch check ---------------------------------------------

// cmd_parcheck runs the same linear through the pool and single-threaded and
// reports the first divergent output row. A mismatch means a worker produced a
// row range it should not have, which is a bug in the pool, not in the kernel.
cmd_parcheck :: proc(args: []string) {
	tensor := "enc.blocks.0.ff1.linear1.weight"
	workers := 8
	if len(args) > 3 { workers, _ = strconv.parse_int(args[3]) }
	if len(args) > 4 { tensor = args[4] }

	g, err := phonon.gguf_load(args[2])
	if err != "" { fmt.eprintln(err); os.exit(1) }
	defer phonon.gguf_destroy(&g)

	t, ok := g.tensors[tensor]
	if !ok { fmt.eprintln("no tensor", tensor); os.exit(1) }

	pool := phonon.thread_pool_make(workers)
	defer phonon.thread_pool_free(pool)
	nthreads := pool == nil ? 1 : len(pool.threads)

	// 13 frames: an odd count, so the paired kernel runs and the single-frame
	// tail does too.
	frames := 13
	cols := int(t.dims[0])
	rows := int(t.dims[1])
	x := phonon.mat_make(frames, cols)
	ref := phonon.mat_make(frames, rows)
	got := phonon.mat_make(frames, rows)
	defer phonon.mat_destroy(&x)
	defer phonon.mat_destroy(&ref)
	defer phonon.mat_destroy(&got)
	for i in 0 ..< len(x.data) { x.data[i] = f32(math.sin(f64(i) * 0.0017)) }

	phonon.lin_rows(ref, x, t, nil)

	bad := 0
	for it in 0 ..< 64 {
		for &v in got.data { v = 0 }
		phonon.lin_rows_par(got, x, t, nil, pool, nthreads)
		// Exact bits, not a tolerance: the pooled path must do the same
		// arithmetic as the serial one, so any drift means a kernel is
		// summing different operands in a different order.
		worst: f32 = 0
		worst_at := -1
		for i in 0 ..< len(ref.data) {
			if transmute(u32)got.data[i] != transmute(u32)ref.data[i] {
				d := abs(got.data[i] - ref.data[i])
				if worst_at < 0 || d > worst { worst = d; worst_at = i }
			}
		}
		if worst_at >= 0 {
			bad += 1
			if bad <= 3 {
				fmt.printf("iter %d worst %.6e at flat %d (row %d frame %d), zero_norm=%.4f\n",
					it, worst, worst_at, worst_at %% rows, worst_at / rows,
					f64(norm(got.data)) / f64(norm(ref.data)))
			}
		}
	}
	fmt.printf("PARCHECK threads=%d tensor=%s rows=%d bad=%d/64\n", nthreads, tensor, rows, bad)
}

norm :: proc(v: []f32) -> f32 {
	s: f32 = 0
	for x in v { s += x * x }
	return math.sqrt(s)
}

// --- tensor inventory ------------------------------------------------------

cmd_types :: proc(args: []string) {
	if len(args) < 3 { usage() }
	g, gerr := phonon.gguf_load(args[2], context.allocator)
	if gerr != "" { fmt.eprintln("gguf:", gerr); os.exit(1) }
	filter := len(args) > 3 ? args[3] : ""
	nonf32 := 0
	for name in g.order {
		t := g.tensors[name]
		if filter != "" {
			if !strings.contains(name, filter) { continue }
			nonf32 += 1
			fmt.printf("%-46s %-6v n_dims=%d ne=[%d,%d,%d,%d]\n", name, t.type, t.n_dims, t.dims[0], t.dims[1], t.dims[2], t.dims[3])
			continue
		}
		if t.type == .F32 { continue }
		nonf32 += 1
		fmt.printf("%-46s %-6v ne=[%d,%d]\n", name, t.type, t.dims[0], t.dims[1])
	}
	fmt.printf("tensors %d, non-f32 %d\n", len(g.order), nonf32)
}

// --- parity ----------------------------------------------------------------

Dump :: struct {
	rows, cols: int,
	data:       []f32,
}

dumps: map[string]Dump
ours_dir: string
compared: int
worst: f64

// json_shape pulls "shape": [a, b] out of the reference json without a full
// parser: the dumps always carry a flat two-element shape.
json_shape :: proc(path: string) -> (rows, cols: int, ok: bool) {
	text, err := os.read_entire_file(path, context.temp_allocator)
	if err != nil { return 0, 0, false }
	s := string(text)
	i := strings.index(s, "\"shape\"")
	if i < 0 { return 0, 0, false }
	open_b := strings.index(s[i:], "[")
	close_b := strings.index(s[i:], "]")
	if open_b < 0 || close_b < 0 { return 0, 0, false }
	body := s[i + open_b + 1:][:close_b - open_b - 1]
	parts := strings.split(body, ",", context.temp_allocator)
	if len(parts) == 1 {
		v, _ := strconv.parse_int(strings.trim_space(parts[0]))
		return 1, v, true
	}
	r, _ := strconv.parse_int(strings.trim_space(parts[0]))
	c, _ := strconv.parse_int(strings.trim_space(parts[1]))
	return r, c, true
}

trace_sink :: proc(name: string, m: phonon.Mat) {
	d, ok := dumps[name]
	if !ok {
		fmt.printf("%-24s (no reference)\n", name)
		return
	}
	if d.rows != m.rows || d.cols != m.cols {
		fmt.printf("%-24s SHAPE ref[%d,%d] ours[%d,%d] (skipped)\n", name, d.rows, d.cols, m.rows, m.cols)
		return
	}
	if ours_dir != "" {
		raw := mem.slice_data_cast([]byte, m.data)
		_ = os.write_entire_file(fmt.aprintf("%s/%s.f32", ours_dir, name), raw)
	}
	n := min(len(d.data), len(m.data))
	max_diff, sum_diff, sum_ref: f64 = 0, 0, 0
	for i in 0 ..< n {
		diff := math.abs(f64(m.data[i]) - f64(d.data[i]))
		if diff > max_diff { max_diff = diff }
		sum_diff += diff
		sum_ref += math.abs(f64(d.data[i]))
	}
	rel := sum_ref == 0 ? 0 : sum_diff / sum_ref
	compared += 1
	if max_diff > worst { worst = max_diff }
	fmt.printf(
		"%-24s shape ref[%d,%d] ours[%d,%d]  max|d| %.3e  mean|d| %.3e  sum_rel %.3e\n",
		name, d.rows, d.cols, m.rows, m.cols, max_diff, sum_diff / f64(n), rel,
	)
}

load_dumps :: proc(ddir: string) {
	dir, derr := os.open(ddir)
	if derr != nil { fmt.eprintln("cannot open dump dir", ddir); os.exit(1) }
	entries, _ := os.read_directory(dir, -1, context.temp_allocator)
	os.close(dir)
	for e in entries {
		if !strings.has_suffix(e.name, ".json") { continue }
		base := strings.trim_suffix(e.name, ".json")
		rows, cols, ok := json_shape(strings.concatenate({ddir, "/", e.name}, context.temp_allocator))
		if !ok { continue }
		raw, rerr := os.read_entire_file(strings.concatenate({ddir, "/", base, ".f32"}, context.temp_allocator), context.allocator)
		if rerr != nil { continue }
		data := make([]f32, len(raw) / 4)
		copy(mem.slice_data_cast([]byte, data), raw)
		dumps[base] = Dump{rows = rows, cols = cols, data = data}
	}
	fmt.printf("reference tensors: %d\n", len(dumps))
}

// cmd_parity_enc is the encoder-only numeric gate: it feeds the *reference*
// mel plane (enc.mel.in from the dump dir) straight through phonon.encode and
// diffs every encoder stage. The reference dumps were produced at 84 mel
// frames (transcribe.cpp pads the frontend), the Odin frontend emits 106, so
// the end-to-end parity run can only compare shapes here; this verb removes
// that difference from the comparison so kernel and layer changes can be gated
// on numbers instead of on a transcript whose decoder is still WIP.
cmd_parity_enc :: proc(args: []string) {
	if len(args) < 4 { usage() }
	model, ddir := args[2], args[3]
	ours_dir = strings.concatenate({ddir, "/../ours"})
	load_dumps(ddir)

	mel, ok := dumps["enc.mel.in"]
	if !ok { fmt.eprintln("no enc.mel.in in", ddir); os.exit(1) }

	workers := 1
	if len(args) > 4 { workers, _ = strconv.parse_int(args[4]) }
	e, err := phonon.engine_open(model, workers)
	if err != "" { fmt.eprintln("engine:", err); os.exit(1) }
	defer phonon.engine_close(&e)
	phonon.set_trace(&e, trace_sink)

	start := time.tick_now()
	enc := phonon.encode(&e.enc, mel.data, mel.rows, mel.cols, context.temp_allocator)
	el := ms_since(start)
	_ = enc
	fmt.printf("PARITY_ENC frames=%d ms=%.2f compared=%d worst=%.4e\n", mel.cols, el, compared, worst)
}

cmd_parity :: proc(args: []string) {
	if len(args) < 5 { usage() }
	model, wav, ddir := args[2], args[3], args[4]
	ours_dir = strings.concatenate({ddir, "/../ours"})
	load_dumps(ddir)

	raw, _, ok := wav_read_mono16k(wav)
	if !ok { fmt.eprintln("cannot read", wav); os.exit(1) }
	samples, trimmed_ms := trim_trailing_silence(raw)
	fmt.printf("samples %d (of %d, trimmed %d ms)\n", len(samples), len(raw), trimmed_ms)

	e, err := phonon.engine_open(model, 1)
	if err != "" { fmt.eprintln("engine:", err); os.exit(1) }
	defer phonon.engine_close(&e)
	phonon.set_trace(&e, trace_sink)

	res := phonon.transcribe(&e, samples)
	fmt.printf("text: %q\n", res.text)
	fmt.printf("tokens:")
	for t in res.tokens { fmt.printf(" %d", t) }
	fmt.printf("\ncompared %d stages, worst max|d| %.3e\n", compared, worst)
}
