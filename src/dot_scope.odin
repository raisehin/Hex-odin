package main

// Pure-Odin port of hex 3.4.0's Dot scope simulation, translated from the
// byte-verified V port in hex-v/v/dotscope.v (which matches the Rust oracle).
// f32 libm entry points are used so the packed u16 dot buffer stays identical.

import "core:c/libc"

foreign import ucrt "system:libucrt.lib"
@(default_calling_convention="c")
foreign ucrt {
	@(link_name="sinf")   m_sinf   :: proc(x: f32) -> f32 ---
	@(link_name="cosf")   m_cosf   :: proc(x: f32) -> f32 ---
	@(link_name="tanf")   m_tanf   :: proc(x: f32) -> f32 ---
	@(link_name="expf")   m_expf   :: proc(x: f32) -> f32 ---
	@(link_name="powf")   m_powf   :: proc(x, y: f32) -> f32 ---
	@(link_name="sqrtf")  m_sqrtf  :: proc(x: f32) -> f32 ---
	@(link_name="tanhf")  m_tanhf  :: proc(x: f32) -> f32 ---
	@(link_name="floorf") m_floorf :: proc(x: f32) -> f32 ---
	@(link_name="roundf") m_roundf :: proc(x: f32) -> f32 ---
	@(link_name="log10f") m_log10f :: proc(x: f32) -> f32 ---
	@(link_name="fmodf")  m_fmodf  :: proc(x, y: f32) -> f32 ---
}

PI :: f32(3.14159265358979323846)
TAU :: f32(6.28318530717958647692)
STEP_H :: f32(1.0) / f32(240.0)
HISTORY :: 512
SAMPLES_PER_COLUMN :: 3
SPEEDS: [4]f32 = {0.8, 1.25, 1.85, 2.6}
BASE_AMP: [4]f32 = {0.55, 0.42, 0.3, 0.2}
SPATIAL: [4]f32 = {1.4, 2.4, 3.8, 5.6}
START_WIDTH :: f32(30.0)
ENTER_TIME :: f32(0.42)
WIPE :: f32(0.3)
PASS_INSET :: f32(9.0)
PHOSPHOR_HEAD :: f32(0.15)
BREATH_PERIOD :: f32(5.5)
HIDDEN_SCALE :: f32(0.9)
HIDDEN_SOFTNESS :: f32(3.0)
WORK_HELIX :: f32(2.6)
FLOW_SPACING_K :: f32(0.7)
CARRIER_REST :: f32(0.14)
RING_BIRTH :: f32(0.15)
BEAM_FADE :: f32(0.14)
MAX_COLUMNS :: 96
MAX_ROWS :: 15
MAX_DOTS :: 1024
PLATE_HEIGHT :: f32(26.0)
AUTO_MIN_SPAN :: f32(16.0)
AUTO_RANGE :: f32(24.0)
AUTO_ROOM_MARGIN :: f32(5.0)
AUTO_ATTACK :: f32(0.04)
ROOM_FALL :: f32(0.3)
ROOM_RISE :: f32(6.0)
MAX_F32 :: f32(3.4028234663852886e38)

clamp32 :: proc(x, lo, hi: f32) -> f32 {
	if x < lo { return lo }
	if x > hi { return hi }
	return x
}
fmin32 :: proc(a, b: f32) -> f32 { return a < b ? a : b }
fmax32 :: proc(a, b: f32) -> f32 { return a > b ? a : b }
fmin_int :: proc(a, b: int) -> int { return a < b ? a : b }
fabs32 :: proc(x: f32) -> f32 { return x < 0 ? -x : x }

rem_euclid32 :: proc(x, y: f32) -> f32 {
	r := m_fmodf(x, y)
	if r < 0 { return r + (y < 0 ? -y : y) }
	return r
}
smoothstep :: proc(a, b, x: f32) -> f32 {
	t := clamp32((x - a) / (b - a), 0, 1)
	return t * t * (3 - 2 * t)
}
bump :: proc(d, w: f32) -> f32 { return m_expf(-(d / w) * (d / w)) }
hermite :: proc(start, speed, end, t_in: f32) -> f32 {
	t := clamp32(t_in, 0, 1)
	t2 := t * t
	t3 := t * t * t
	return (2 * t3 - 3 * t2 + 1) * start + (t3 - 2 * t2 + t) * WIPE * speed + (3 * t2 - 2 * t3) * end
}
pass_position :: proc(p: f32) -> f32 { return 0.5 - 0.5 * m_cosf(PI * clamp32(p, 0, 1)) }

demo_level :: proc(time: f32) -> f32 {
	phrase := m_floorf(time / f32(2.4))
	t := time - phrase * f32(2.4)
	if t > 1.7 { return 0.04 }
	shape := m_powf(m_sinf(PI * t / f32(1.7)), f32(0.4))
	rate := TAU * (f32(4.3) + f32(0.6) * m_sinf(phrase * f32(1.7)))
	syllable := f32(0.5) + f32(0.5) * m_sinf(rate * t + phrase)
	stress := f32(0.75) + f32(0.25) * m_sinf(TAU * f32(1.1) * t + f32(2.0) * phrase)
	return clamp32(f32(0.08) + f32(0.82) * shape * m_powf(syllable, f32(1.6)) * stress, 0, 1)
}
demo_loudness :: proc(time: f32) -> f32 { return -58 + 28 * demo_level(time) }
loudness_dbfs :: proc(rms: f32) -> f32 { return 20 * m_log10f(fmax32(rms, 0) + f32(1e-6)) }
fixed_level :: proc(loudness: f32) -> f32 { return clamp32((loudness + 40) / 30, 0, 1) }

Spec :: struct { duration, bounce: f32 }
spec_new :: proc(duration, bounce: f32) -> Spec { return Spec{duration, bounce} }
spec_omega :: proc(s: Spec) -> f32 { return TAU / fmax32(s.duration, f32(0.01)) }
spec_zeta :: proc(s: Spec) -> f32 {
	if s.bounce >= 0 { return fmax32(1 - s.bounce, 0) }
	return 1 / fmax32(1 + s.bounce, f32(0.01))
}
REDUCED_FADE :: Spec{f32(0.16), 0}

Spring :: struct { x, v, target: f32 }
spring_at :: proc(x: f32) -> Spring { return Spring{x, 0, x} }
spring_jump :: proc(s: ^Spring, x: f32) { s.x = x; s.v = 0; s.target = x }
spring_step_to :: proc(s: ^Spring, target: f32, spec: Spec, dt: f32) {
	s.target = target
	spring_step(s, spec, dt)
}
spring_step :: proc(s: ^Spring, spec: Spec, dt: f32) {
	if dt <= 0 { return }
	omega := spec_omega(spec)
	zeta := spec_zeta(spec)
	x0 := s.x - s.target
	v0 := s.v
	if zeta < 1 - f32(1e-4) {
		decay := zeta * omega
		omega_d := omega * m_sqrtf(1 - zeta * zeta)
		ex := m_expf(-decay * dt)
		sin_v := m_sinf(omega_d * dt)
		cos_v := m_cosf(omega_d * dt)
		sin_over_d := sin_v / omega_d
		s.x = s.target + ex * (x0 * (cos_v + decay * sin_over_d) + v0 * sin_over_d)
		s.v = ex * (-x0 * omega * omega * sin_over_d + v0 * (cos_v - decay * sin_over_d))
	} else if zeta > 1 + f32(1e-4) {
		root := m_sqrtf(zeta * zeta - 1)
		root_sum := zeta + root
		slow_root := -omega / root_sum
		fast_root := -omega * root_sum
		denom := slow_root - fast_root
		slow_exp := m_expf(slow_root * dt)
		fast_exp := m_expf(fast_root * dt)
		p00 := (-fast_root * slow_exp + slow_root * fast_exp) / denom
		p01 := (slow_exp - fast_exp) / denom
		p10 := slow_root * fast_root * (fast_exp - slow_exp) / denom
		p11 := (slow_root * slow_exp - fast_root * fast_exp) / denom
		s.x = s.target + p00 * x0 + p01 * v0
		s.v = p10 * x0 + p11 * v0
	} else {
		ex := m_expf(-omega * dt)
		c := v0 + omega * x0
		s.x = s.target + (x0 + c * dt) * ex
		s.v = (v0 - omega * c * dt) * ex
	}
}

Processing :: enum { sweep, flow, pulse }
Placement :: enum { below, menu_bar, notch }

Tuning :: struct {
	columns, rows, spacing, width, processing_width, height: f32,
	braid_speed, helix_cycles, gain, response, release, history_speed: f32,
	phosphor, glow, saturation, shimmer_pass, shimmer, transition, bounce: f32,
	auto_level: bool,
	auto_release, noise_floor: f32,
	processing: Processing,
	placement: Placement,
	pretend_notch: bool,
	island_spread: f32,
}

tuning_default :: proc() -> Tuning {
	return Tuning{
		columns = 43, rows = 9, spacing = 2.6, width = 112,
		processing_width = 76, height = PLATE_HEIGHT,
		braid_speed = 1, helix_cycles = 1.6, gain = 1, response = 6.5,
		release = 0.15, history_speed = 150, phosphor = 0.6, glow = 1,
		saturation = 1, shimmer_pass = 0.55, shimmer = 1,
		transition = 0.30, bounce = 0.15, auto_level = true,
		auto_release = 3.0, noise_floor = -60, processing = .flow,
		placement = .below, pretend_notch = false, island_spread = 28,
	}
}

tuning_clamped :: proc(t: Tuning) -> Tuning {
	o := t
	rows := fmin_int(MAX_DOTS / int(o.columns), MAX_ROWS)
	if int(o.rows) > rows { o.rows = f32(rows) }
	return o
}

Stage :: enum { recording, processing, done, failed }
Frame :: struct { stage: Stage, visible: bool, loudness: f32, reduce_motion: bool }
Pose :: struct {
	width, height, opacity, scale, softness, cool, fail, glow, grid, live,
	reveal, level, clock, done: f32,
}

Voice :: struct { level, reservoir, energy, kick, kick_velocity, band: f32 }
voice_step :: proc(v: ^Voice, raw, h: f32) {
	previous := v.level
	time := raw > v.level ? f32(0.035) : f32(0.095)
	v.level += (raw - v.level) * (1 - m_expf(-h / time))
	rise := fmax32((v.level - previous) / h, 0)
	v.reservoir += (1.75 * v.level * v.level - v.reservoir / 1.8) * h
	v.energy = 1 - m_expf(-v.reservoir * 1.45)
	w := TAU * f32(1.3)
	z := f32(0.22)
	v.kick_velocity += (-w * w * v.kick - 2 * z * w * v.kick_velocity + 15 * rise + 2.2 * v.level) * h
	v.kick += v.kick_velocity * h
	time2 := raw > v.band ? f32(0.025) : f32(0.14)
	v.band += (raw - v.band) * (1 - m_expf(-h / time2))
}

Trace :: struct {
	envelope, amplitude, amplitude_velocity, glow_fast, glow_spring,
	glow_spring_velocity, glow_slow, glow: f32,
	phases: [4]f32,
	history: [HISTORY * 5]f32,
	head: int,
}
trace_new :: proc(phases: [4]f32) -> Trace { return Trace{phases = phases} }
trace_tick :: proc(t: ^Trace, v: ^Voice, live, speed: f32, tuning: Tuning, h: f32) {
	t.envelope += (v.level - t.envelope) * (1 - m_expf(-h / (v.level > t.envelope ? f32(0.022) : tuning.release)))
	w := TAU * tuning.response
	z := f32(0.52)
	t.amplitude_velocity += (-w * w * (t.amplitude - t.envelope) - 2 * z * w * t.amplitude_velocity) * h
	t.amplitude += t.amplitude_velocity * h
	t.glow_fast += (v.level - t.glow_fast) * (1 - m_expf(-h / (v.level > t.glow_fast ? f32(0.045) : f32(0.13))))
	gw := TAU * f32(3.6)
	gz := f32(0.5)
	t.glow_spring_velocity += (-gw * gw * (t.glow_spring - t.glow_fast) - 2 * gz * gw * t.glow_spring_velocity) * h
	t.glow_spring += t.glow_spring_velocity * h
	t.glow_slow += (v.level - t.glow_slow) * (1 - m_expf(-h / (v.level > t.glow_slow ? f32(0.15) : f32(0.7))))
	swell := 0.8 * fmax32(t.glow_spring - 0.25, 0) + 0.04 * fmax32(v.kick, -0.5)
	t.glow = fmin32(fmax32(swell, 0) + 0.42 * t.glow_slow + 0.2 * v.energy, f32(1.15)) * live
	span := 0.6 + 1.1 * fmin32(t.envelope * 1.6, 1)
	for i in 0..<4 {
		t.phases[i] = m_fmodf(t.phases[i] + h * TAU * SPEEDS[i] * span * speed, TAU)
	}
	emphasis := 0.35 + 0.65 * fmin32(v.band / (v.band + 0.06), 2)
	offset := t.head * 5
	amplitude := fmax32(t.amplitude, 0) * live
	t.history[offset] = amplitude
	for c in 0..<4 {
		t.history[offset + 1 + c] = amplitude * BASE_AMP[c] * emphasis
	}
	t.head = (t.head + 1) % HISTORY
}
trace_past :: proc(t: ^Trace, delay: f32, channel: int) -> f32 {
	d := fmin32(delay / STEP_H, f32(HISTORY - 2))
	i := int(m_floorf(d))
	k := d - f32(i)
	a := (t.head + 2 * HISTORY - 1 - i) % HISTORY
	b := (a + HISTORY - 1) % HISTORY
	return t.history[a * 5 + channel] * (1 - k) + t.history[b * 5 + channel] * k
}

Working :: struct { kind: int, offset, spacing, time, period, speed: f32 }
working_light :: proc(w: Working, x, half_width: f32) -> f32 {
	if w.kind == 0 {
		d := rem_euclid32(x - w.offset + w.spacing / 2, w.spacing) - w.spacing / 2
		return bump(d, 7)
	}
	d := x < 0 ? -x : x
	light := f32(0)
	age := rem_euclid32(w.time, w.period)
	for age * w.speed < half_width + 12 && age <= w.time + f32(1e-4) {
		radius := age * w.speed
		fade := smoothstep(0, RING_BIRTH, age) * (1 - 0.6 * smoothstep(0.3 * half_width, half_width, radius))
		light += bump(d - radius, 6) * fade
		age += w.period
	}
	return fmin32(light, 1)
}

Scan :: struct {
	helix, decay, tail, grid: f32,
	kind: int,
	reach, head, unfold, iron, front: f32,
	working: Working,
	swell, half_width, beam, beam_at, field, glow: f32,
}
scan_at :: proc(s: Scan, x: f32) -> (f32, f32, f32) {
	switch s.kind {
	case 0:
		d := x < 0 ? -x : x
		drawn := s.unfold * (1 - smoothstep(fmax32(s.reach - 14, 0), fmax32(s.reach, f32(0.001)), d))
		amplitude := drawn * (1 - smoothstep(fmax32(s.reach - 22, 0), fmax32(s.reach - 3, f32(0.001)), d))
		beam := s.head * bump(d - s.reach, 6)
		swell := 1.8 * s.unfold * (s.head / 2.8) * bump(d - fmax32(s.reach - 14, 0), 14)
		return amplitude, amplitude + beam, swell
	case 1:
		return s.iron, s.iron + s.head * bump(x - s.front, 5), 0
	case 2:
		light := working_light(s.working, x, s.half_width)
		return 1, 1 + s.head * light, s.swell * light
	case:
		d := x < 0 ? -x : x
		keep := s.field * smoothstep(s.front - 12, s.front - 2, d)
		light := working_light(s.working, x, s.half_width)
		return keep, keep * (1 + s.glow * light) + s.head * bump(d - s.front, 6) + s.beam * bump(x - s.beam_at, 5), keep * s.swell * light
	}
}

AutoLevel :: struct { peak, room: f32 }
auto_level_default :: proc() -> AutoLevel { return AutoLevel{-30, -55} }
auto_level_step :: proc(a: ^AutoLevel, loudness, h, release, noise_floor: f32) -> f32 {
	db := fmax32(loudness, -120)
	room_time := db < a.room ? ROOM_FALL : ROOM_RISE
	a.room += (db - a.room) * (1 - m_expf(-h / fmax32(room_time, f32(1e-3))))
	lowest := fmax32(a.room, noise_floor) + AUTO_MIN_SPAN
	if db > a.peak {
		a.peak += (db - a.peak) * (1 - m_expf(-h / fmax32(AUTO_ATTACK, f32(1e-3))))
	} else {
		a.peak += (lowest - a.peak) * (1 - m_expf(-h / fmax32(release, f32(1e-3))))
	}
	a.peak = fmax32(a.peak, lowest)
	return auto_level_value(a, db, noise_floor)
}
auto_level_value :: proc(a: ^AutoLevel, loudness, noise_floor: f32) -> f32 {
	low := fmax32(fmax32(a.room, noise_floor) + AUTO_ROOM_MARGIN, a.peak - AUTO_RANGE)
	return clamp32((loudness - low) / fmax32(a.peak - low, 1), 0, 1)
}

Scope :: struct {
	tuning: Tuning,
	stage: Stage,
	elapsed, speed_at_done, done_from, flow_offset, flow_turn: f32,
	auto: AutoLevel,
	level: f32,
	voices: [2]Voice,
	traces: [2]Trace,
	live, carrier, time: f32,
	steps: u64,
	pending: f32,
	reduce_motion: bool,
	columns, rows: int,
	head, tail: [2][]f32,
	dots: [][2]f32,
	ys, brightness: [2][]f32,
	grid: f32,
	width, opacity, scale, softness, cool, fail: Spring,
}

scope_new :: proc(tuning: Tuning) -> ^Scope {
	s := new(Scope)
	s.tuning = tuning_clamped(tuning)
	s.stage = .recording
	s.auto = auto_level_default()
	s.traces[0] = trace_new([4]f32{0.1, 1.7, 3.3, 4.6})
	s.traces[1] = trace_new([4]f32{2.4, 0.6, 5.1, 1.9})
	s.width = spring_at(START_WIDTH)
	s.opacity = spring_at(0)
	s.scale = spring_at(HIDDEN_SCALE)
	s.softness = spring_at(HIDDEN_SOFTNESS)
	s.cool = spring_at(0)
	s.fail = spring_at(0)
	scope_resize(s)
	return s
}

scope_resize :: proc(s: ^Scope) {
	columns := int(s.tuning.columns)
	rows := int(s.tuning.rows)
	if columns == s.columns && rows == s.rows { return }
	s.columns = columns
	s.rows = rows
	dots := columns * rows
	samples := (columns - 1) * SAMPLES_PER_COLUMN + 1
	for k in 0..<2 {
		s.head[k] = make([]f32, dots)
		s.tail[k] = make([]f32, dots)
		s.ys[k] = make([]f32, samples)
		s.brightness[k] = make([]f32, samples)
	}
	s.dots = make([][2]f32, dots)
}

scope_pose :: proc(s: ^Scope) -> Pose {
	glow_sum := f32(0)
	for k in 0..<2 { glow_sum += s.traces[k].glow }
	return Pose{
		width = fmax32(s.width.x, s.tuning.height), height = s.tuning.height,
		opacity = clamp32(s.opacity.x, 0, 1), scale = fmax32(s.scale.x, f32(0.01)),
		softness = clamp32(s.softness.x, 0, HIDDEN_SOFTNESS),
		cool = clamp32(s.cool.x, 0, 1), fail = clamp32(s.fail.x, 0, 1),
		glow = glow_sum / 2 * s.tuning.glow, grid = s.grid, live = s.live,
		reveal = fmax32((s.scale.x - HIDDEN_SCALE) / (1 - HIDDEN_SCALE), 0),
		level = clamp32(s.voices[0].level, 0, 1),
		clock = s.reduce_motion ? 0 : s.time,
		done = s.stage == .done ? 1 : 0,
	}
}

scope_advance :: proc(s: ^Scope, dt: f32, frame: Frame) {
	if frame.stage != s.stage {
		if frame.stage == .done { scope_arm_done(s) }
		if frame.stage == .processing { s.flow_offset = -s.width.x / 2 - 4 }
		s.stage = frame.stage
		s.elapsed = 0
	}
	s.reduce_motion = frame.reduce_motion
	s.pending += clamp32(dt, 0, 0.25)
	for s.pending >= STEP_H {
		s.pending -= STEP_H
		scope_step(s, frame)
	}
}

scope_step :: proc(s: ^Scope, frame: Frame) {
	h := STEP_H
	tuning := s.tuning
	recording := s.stage == .recording && frame.visible
	level := f32(0)
	if recording {
		s.level = tuning.auto_level ? auto_level_step(&s.auto, frame.loudness, h, tuning.auto_release, tuning.noise_floor) : fixed_level(frame.loudness)
		level = clamp32(s.level * tuning.gain, 0, 1)
	} else {
		s.level = 0
	}
	voice_step(&s.voices[0], level, h)
	voice_step(&s.voices[1], level * 0.85, h)
	live_target := recording ? f32(1) : f32(0)
	s.live += (live_target - s.live) * (1 - m_expf(-h / f32(0.55)))
	speed := frame.reduce_motion ? f32(0) : tuning.braid_speed
	for k in 0..<2 { trace_tick(&s.traces[k], &s.voices[k], s.live, speed, tuning, h) }
	energy := fmax32(s.voices[0].energy, s.voices[1].energy)
	working := s.stage == .processing || s.stage == .done
	flowing := working && tuning.processing == .flow
	s.flow_turn += ((flowing ? f32(1) : f32(0)) - s.flow_turn) * (1 - m_expf(-h / f32(0.25)))
	if working { s.flow_offset += scope_flow_speed(s) * h }
	rest := CARRIER_REST + 0.22 * energy
	turn := rest + (scope_flow_turn_hz(s) - rest) * s.flow_turn
	s.carrier = m_fmodf(s.carrier + h * TAU * turn * speed, TAU)
	width := tuning.width
	switch s.stage {
	case .recording: width = tuning.width
	case .processing, .done: width = tuning.processing_width
	case .failed: width = s.width.target
	}
	cool_target := (s.stage == .processing || s.stage == .done) ? f32(1) : f32(0)
	fail_target := s.stage == .failed ? f32(1) : f32(0)
	shown := frame.visible ? f32(1) : f32(0)
	fade := spec_new(tuning.transition, 0)
	bouncy := spec_new(tuning.transition, tuning.bounce)
	if frame.reduce_motion {
		spring_jump(&s.width, width)
		spring_jump(&s.scale, 1)
		spring_jump(&s.softness, 0)
		spring_step_to(&s.opacity, shown, REDUCED_FADE, h)
		spring_step_to(&s.cool, cool_target, REDUCED_FADE, h)
		spring_step_to(&s.fail, fail_target, REDUCED_FADE, h)
	} else {
		spring_step_to(&s.width, width, bouncy, h)
		spring_step_to(&s.opacity, shown, fade, h)
		spring_step_to(&s.scale, frame.visible ? f32(1) : HIDDEN_SCALE, bouncy, h)
		spring_step_to(&s.softness, frame.visible ? f32(0) : HIDDEN_SOFTNESS, fade, h)
		spring_step_to(&s.cool, cool_target, fade, h)
		spring_step_to(&s.fail, fail_target, fade, h)
	}
	s.elapsed += h
	s.time += h
	s.steps += 1
	if s.steps % 2 == 0 { scope_beam(s, 2 * h) }
}

scope_work_period :: proc(s: ^Scope) -> f32 { return 2 * fmax32(s.tuning.shimmer_pass, f32(0.05)) }
scope_flow_spacing :: proc(s: ^Scope) -> f32 { return FLOW_SPACING_K * s.tuning.processing_width }
scope_flow_speed :: proc(s: ^Scope) -> f32 { return scope_flow_spacing(s) / scope_work_period(s) }
scope_flow_turn_hz :: proc(s: ^Scope) -> f32 { return scope_flow_speed(s) * s.tuning.helix_cycles / fmax32(s.width.x, 1) }
scope_pulse_speed :: proc(s: ^Scope) -> f32 { return (s.tuning.processing_width / 2 + 12) / (1.4 * scope_work_period(s)) }
scope_working :: proc(s: ^Scope, time: f32) -> Working {
	if s.tuning.processing == .pulse {
		return Working{kind = 1, time = time, period = scope_work_period(s), speed = scope_pulse_speed(s)}
	}
	return Working{kind = 0, offset = s.flow_offset, spacing = scope_flow_spacing(s)}
}
scope_arm_done :: proc(s: ^Scope) {
	s.speed_at_done = 1.5 * scope_pulse_speed(s)
	s.done_from = s.stage == .processing ? s.elapsed : 0
}
scope_process_head :: proc(s: ^Scope, elapsed: f32) -> f32 {
	pass := fmax32(s.tuning.shimmer_pass, f32(0.05))
	return (2.2 + fabs32(m_sinf(PI * (elapsed / pass + 0.5)))) * smoothstep(0, 0.14, elapsed) * s.tuning.shimmer
}
scope_process_front :: proc(s: ^Scope, elapsed, width: f32) -> f32 {
	pass := fmax32(s.tuning.shimmer_pass, f32(0.05))
	position := 0.5 - 0.5 * m_cosf(PI * (elapsed / pass + 0.5))
	return -width / 2 + PASS_INSET + (width - 2 * PASS_INSET) * position
}

scope_scan :: proc(s: ^Scope, width: f32) -> Scan {
	e := s.elapsed
	if s.stage == .failed || s.reduce_motion {
		return Scan{helix = 1, decay = 1, tail = 1, grid = 1, reach = 1e4, head = 0, unfold = 1}
	}
	if s.stage == .recording {
		return Scan{
			kind = 0, helix = 1, decay = 1, tail = 1,
			grid = fmin32(e / 0.3, 1),
			reach = (width / 2 + 12) * pass_position(e / ENTER_TIME),
			head = 2.8 * (1 - smoothstep(0.08, ENTER_TIME, e)),
			unfold = smoothstep(0, 0.1, e),
		}
	}
	if s.stage == .processing && s.tuning.processing != .sweep {
		enter := smoothstep(0, 0.35, e)
		return Scan{
			kind = 2, helix = 1 + (WORK_HELIX - 1) * enter, decay = 1, tail = 1, grid = 1,
			working = scope_working(s, e),
			head = 2.5 * smoothstep(0.05, 0.3, e) * s.tuning.shimmer,
			swell = 0.9 * enter, half_width = width / 2,
		}
	}
	if s.stage == .done {
		target := width / 2 + 8
		front := hermite(0, s.speed_at_done, target, e / WIPE)
		sweep := s.tuning.processing == .sweep
		return Scan{
			kind = 3, helix = sweep ? f32(0) : WORK_HELIX,
			decay = 1 - 0.6 * smoothstep(0, 0.12, e),
			tail = 1 - smoothstep(0, 0.15, e),
			grid = 1 - smoothstep(0.2, 0.4, e), front = front,
			head = 2.8 * smoothstep(0, 0.04, e) * (1 - smoothstep(0.22, 0.32, e)),
			beam = sweep && s.done_from > 0 ? scope_process_head(s, s.done_from + e) * (1 - smoothstep(0, BEAM_FADE, e)) : f32(0),
			beam_at = scope_process_front(s, s.done_from + e, width),
			field = sweep ? f32(0) : f32(1), working = scope_working(s, s.done_from + e),
			glow = (sweep ? f32(0) : f32(2.5)) * s.tuning.shimmer * (1 - smoothstep(0, 0.18, e)),
			swell = 0.9, half_width = width / 2,
		}
	}
	iron := 1 - smoothstep(0, 0.28, e)
	return Scan{kind = 1, helix = iron, decay = 1, tail = 1, grid = 1, iron = iron, front = scope_process_front(s, e, width), head = scope_process_head(s, e)}
}

scope_beam :: proc(s: ^Scope, hb: f32) {
	columns := s.columns
	rows := s.rows
	sub := SAMPLES_PER_COLUMN
	samples := (columns - 1) * sub + 1
	pitch := s.tuning.spacing
	width := fmax32(s.width.x, s.tuning.height)
	scan := scope_scan(s, width)
	s.grid = scan.grid
	mid := (f32(rows) - 1) / 2
	half := mid - 0.55
	breath := s.reduce_motion ? f32(0.5) : 0.5 + 0.5 * m_sinf(s.time * TAU / BREATH_PERIOD)
	rest := (0.55 + 0.35 * breath) * (0.4 + 0.6 * s.live) * scan.helix
	half_width := width / 2
	for k in 0..<2 {
		trace := &s.traces[k]
		sign := k == 0 ? f32(1) : f32(-1)
		ys := s.ys[k]
		brightness := s.brightness[k]
		for si in 0..<samples {
			x := (f32(si) / f32(sub) - (f32(columns) - 1) / 2) * pitch
			if fabs32(x) > half_width {
				ys[si] = mid
				brightness[si] = 0
				continue
			}
			amplitude, bright, swell := scan_at(scan, x)
			own := fabs32(x)
			xn := (x + half_width) / width
			ends := fmin32(smoothstep(-half_width + 3, -half_width + 17, x), smoothstep(-half_width + 3, -half_width + 17, -x))
			reach := 1 - 0.8 * smoothstep(14, half_width, own)
			delay := own / s.tuning.history_speed
			a := trace_past(trace, delay, 0)
			spread := f32(0)
			for c in 0..<4 {
				spread += trace_past(trace, delay, 1 + c) * m_sinf(TAU * SPATIAL[c] * xn - trace.phases[c])
			}
			helix := (rest + swell + half * 0.55 * a * reach) * m_sinf(TAU * s.tuning.helix_cycles * xn - s.carrier)
			y := half * m_tanhf((helix + half * 0.7 * spread * reach) / half) * ends * amplitude
			ys[si] = mid - sign * y
			spatial_amt := (0.3 + 0.7 * fmin32(a * 1.8, 1)) * (0.45 + 0.55 * ends)
			brightness[si] = spatial_amt * amplitude + 0.36 * fmax32(bright - amplitude, 0)
		}
		for si in 0..<samples {
			prev := si == 0 ? ys[0] : ys[si - 1]
			next_i := si + 1 > samples - 1 ? samples - 1 : si + 1
			slope := (ys[next_i] - prev) * f32(sub) / 2
			brightness[si] *= 1 - 0.55 + 0.55 / m_sqrtf(1 + 0.22 * slope * slope)
		}
	}
	head_keep := m_expf(-hb / (PHOSPHOR_HEAD * scan.decay))
	tail_keep := m_expf(-hb / (s.tuning.phosphor * scan.decay))
	dx := 1 / f32(sub)
	for i in 0..<columns {
		s0 := (i > 3 ? i - 3 : 0) * sub
		s1 := fmin_int((i + 3) * sub, samples - 1)
		for j in 0..<rows {
			n := j * columns + i
			now := f32(0)
			total := f32(0)
			lit := [2]f32{0, 0}
			for k in 0..<2 {
				ys := s.ys[k]
				best := MAX_F32
				nearest := i * sub
				for si in s0..<s1 {
					ay := ys[si]
					dy := ys[si + 1] - ay
					px := f32(i) - f32(si) * dx
					py := f32(j) - ay
					t := clamp32((px * dx + py * dy) / (dx * dx + dy * dy), 0, 1)
					ex := px - t * dx
					ey := py - t * dy
					d2 := ex * ex + ey * ey
					if d2 < best { best = d2; nearest = si }
				}
				bright := s.brightness[k][nearest]
				on_beam := m_expf(-best / 0.25)
				instant := fmin32(on_beam + 0.5 * m_expf(-m_sqrtf(best) / 0.8), f32(1.2)) * bright
				core := on_beam * bright
				s.head[k][n] = fmax32(s.head[k][n] * head_keep, core)
				s.tail[k][n] = fmax32(s.tail[k][n] * tail_keep, core)
				light := fmax32(fmax32(instant, s.head[k][n]), 0.55 * scan.tail * s.tail[k][n])
				lit[k] = light
				now += instant
				total += light
			}
			power := 1 - (1 - fmin32(lit[0], 1)) * (1 - fmin32(lit[1], 1))
			s.dots[n] = [2]f32{clamp32(power, 0, 1), fmin32(now / fmax32(total, f32(1e-4)), 1)}
		}
	}
}

pack_dots :: proc(dots: [][2]f32) -> []u16 {
	out := make([]u16, len(dots) * 2)
	for d, i in dots {
		for c in 0..<2 {
			v := clamp32(d[c], 0, 1) * 65535
			out[i * 2 + c] = u16(m_roundf(v))
		}
	}
	return out
}

// --- replay ---------------------------------------------------------------

Phase :: enum { hidden, recording, transcribing, completed, cancelled, failed }
Indicator :: struct {
	phase: Phase,
	capturing: bool,
	pending_jobs: int,
	phase_started, working_since: f64,
	completion_pending: bool,
	scope: ^Scope,
	stage: Stage,
	tuning: Tuning,
	loudness, demo_time: f32,
	demo_voice: bool,
}

indicator_new :: proc(tuning: Tuning) -> ^Indicator {
	m := new(Indicator)
	m.phase = .hidden
	m.scope = scope_new(tuning)
	m.stage = .recording
	m.tuning = tuning
	return m
}

indicator_visible_duration :: proc(p: Phase) -> f64 {
	#partial switch p {
	case .completed: return 0.100
	case .cancelled: return 0.0
	case .failed: return 1.2
	case:
		return -1.0
	}
}
phase_stage :: proc(p: Phase) -> (Stage, bool) {
	#partial switch p {
	case .recording: return .recording, true
	case .transcribing: return .processing, true
	case .completed: return .done, true
	case .failed: return .failed, true
	case:
		return .recording, false
	}
}
indicator_show_pipeline :: proc(m: ^Indicator, now: f64) {
	if m.capturing {
		if m.phase != .recording { m.phase = .recording; m.phase_started = now }
		return
	}
	if m.pending_jobs > 0 {
		if m.phase != .transcribing { m.working_since = now }
		m.phase = .transcribing
		m.phase_started = now
		m.completion_pending = false
	}
}
indicator_on_started :: proc(m: ^Indicator, now: f64) {
	m.demo_voice = false
	m.capturing = true
	m.phase = .recording
	m.phase_started = now
	m.completion_pending = false
	m.loudness = -120
	m.stage = .recording
	m.scope = scope_new(m.tuning)
}
indicator_on_submitted :: proc(m: ^Indicator, now: f64) {
	m.capturing = false
	m.pending_jobs = 1
	indicator_show_pipeline(m, now)
}
indicator_on_completed :: proc(m: ^Indicator, now: f64) {
	if m.pending_jobs == 0 { return }
	m.pending_jobs = 0
	if m.capturing { indicator_show_pipeline(m, now); return }
	if m.phase != .transcribing { m.phase = .transcribing; m.working_since = now }
	m.completion_pending = true
}
indicator_step :: proc(m: ^Indicator, now: f64, dt: f32) -> DotFrame {
	d := clamp32(dt, f32(1) / 240, f32(1) / 20)
	if m.completion_pending && now - m.working_since >= 0.200 {
		m.phase = .completed
		m.phase_started = now
		m.completion_pending = false
	}
	elapsed := now - m.phase_started
	vd := indicator_visible_duration(m.phase)
	visible := m.phase != .hidden && !(vd >= 0 && elapsed >= vd)
	if st, ok := phase_stage(m.phase); ok { m.stage = st }
	loud := m.loudness
	if m.demo_voice {
		m.demo_time += d
		loud = demo_loudness(m.demo_time)
	}
	scope_advance(m.scope, d, Frame{stage = m.stage, visible = visible, loudness = loud, reduce_motion = false})
	return DotFrame{
		pose = scope_pose(m.scope), columns = m.scope.columns, rows = m.scope.rows,
		dots = pack_dots(m.scope.dots),
	}
}

DotFrame :: struct {
	pose: Pose,
	columns: int,
	rows: int,
	dots: []u16,
}
