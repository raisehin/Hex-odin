package main

// CPU port of hex 3.4.0's Dot scope fragment shader, `dot_scope_fragment`
// (oracle/rune_hud.wgsl:283-343), for the plate path the HUD draws: a plate
// whose middle and fit come from the layout, no island, and no ears.
//
// Every expression keeps the shader's operand order and f32 width: multiply
// and add are never contracted, and tanh/pow/sqrt/floor/round are the C float
// entry points, so a pixel's colour is the number the GPU computes. The
// uniforms are `DotUniforms` (src/rune_hud.rs:687-712).

import "core:c/libc"

// What one frame draws: the window, the plate's layout and pose, the dot grid,
// and the grid's u16 `[power, fresh]` pairs row by row.
RasterScene :: struct {
	width, height:    int, // device pixels
	backing_scale:    f32, // device pixels per point
	dark:             bool,
	pitch:            f32, // 2.6
	columns, rows:    int,
	dots:             []u16, // columns*rows*2, [power, fresh] per dot
	layout_center_x, layout_center_y, layout_fit: f32, // plate middle in pt, fit 1
	pose_width, pose_height, pose_opacity, pose_scale, pose_softness, pose_cool, pose_fail,
	pose_glow, pose_grid, pose_live, pose_reveal, pose_level, pose_clock,
	pose_done: f32,
}

// Dot scope's colours for one frame, sRGB
// (src/rune_hud/dot_scope/scope.rs:905-970): `Palette`, with the plate colours
// already RGBA.
RasterPalette :: struct {
	base, core: RasterRgb,
	plate, rim, shadow: RasterRgba,
}

RasterRgb :: struct { r, g, b: f32 }

RasterRgba :: struct { r, g, b, a: f32 }

RasterVec2 :: struct { x, y: f32 }

// A dot's shader terms that depend only on the dot. The fragment recomputes
// them from the same operands for every pixel it touches, so one table per
// frame leaves every value bit for bit what it was.
RasterDot :: struct {
	center:     RasterVec2,
	power, mask, alpha, dot_radius: f32,
	color:      RasterRgb,
}

raster_sub :: proc(a, b: RasterVec2) -> RasterVec2 {
	return RasterVec2{a.x - b.x, a.y - b.y}
}

raster_length :: proc(a: RasterVec2) -> f32 {
	return libc.sqrtf(a.x * a.x + a.y * a.y)
}

// `sstep`: `smoothstep` spelled out because `coverage` relies on descending
// edges.
raster_sstep :: proc(edge0, edge1, x: f32) -> f32 {
	t := clamp((x - edge0) / (edge1 - edge0), 0, 1)
	return t * t * (3 - 2 * t)
}

raster_rounded_box :: proc(point, half_size: RasterVec2, radius: f32) -> f32 {
	qx := abs(point.x) - half_size.x + radius
	qy := abs(point.y) - half_size.y + radius
	return raster_length(RasterVec2{max(qx, 0), max(qy, 0)}) + min(max(qx, qy), 0) - radius
}

raster_coverage :: proc(distance, softness: f32) -> f32 {
	return raster_sstep(softness, -softness, distance)
}

raster_lerp :: proc(a, b, t: f32) -> f32 {
	return a + (b - a) * t
}

raster_mix3 :: proc(a, b: RasterRgb, t: f32) -> RasterRgb {
	return RasterRgb{a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t}
}

// `composite`: premultiplied source over premultiplied destination.
raster_composite :: proc(destination: ^RasterRgba, color: RasterRgb, alpha_in: f32) {
	alpha := clamp(alpha_in, 0, 1)
	destination^ = RasterRgba {
		r   = color.r * alpha + destination.r * (1 - alpha),
		g   = color.g * alpha + destination.g * (1 - alpha),
		b   = color.b * alpha + destination.b * (1 - alpha),
		a   = alpha + destination.a * (1 - alpha),
	}
}

raster_fract :: proc(x: f32) -> f32 {
	return x - libc.floorf(x)
}

// Blue-noise-ish dither, +-1/255 of a level, on the pixel grid.
raster_dithered :: proc(result_in: RasterRgba, position: RasterVec2) -> RasterRgba {
	if result_in.a <= 0 {
		return result_in
	}
	px := libc.floorf(position.x)
	py := libc.floorf(position.y)
	n1 := raster_fract(52.9829189 * raster_fract(px * 0.06711056 + py * 0.00583715))
	n2 := raster_fract(
		52.9829189 * raster_fract((px + 17) * 0.06711056 + (py + 31) * 0.00583715),
	)
	dither := ((n1 - n2) / 255) * result_in.a
	return RasterRgba {
		r = clamp(result_in.r + dither, 0, result_in.a),
		g = clamp(result_in.g + dither, 0, result_in.a),
		b = clamp(result_in.b + dither, 0, result_in.a),
		a = result_in.a,
	}
}

// `for_target` at the shader's default `override linear_target: bool = false`:
// the colours stay sRGB-encoded, as they were composed.
raster_for_target :: proc(result: RasterRgba) -> RasterRgba {
	return result
}

// `saturate(color, amount)`: blend towards the luma, then clamp.
raster_saturate :: proc(color: RasterRgb, amount: f32) -> RasterRgb {
	luma := 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b
	return RasterRgb {
		clamp(luma + (color.r - luma) * amount, 0, 1),
		clamp(luma + (color.g - luma) * amount, 0, 1),
		clamp(luma + (color.b - luma) * amount, 0, 1),
	}
}

// `palette(dark, pose, saturation)` at `saturation = 1.0` (Tuning::default()).
raster_palette :: proc(dark: bool, cool, fail: f32) -> RasterPalette {
	you, you_core, agent, agent_core, coral, coral_core: RasterRgb
	if dark {
		you = {0.86, 0.83, 0.77}
		you_core = {0.98, 0.96, 0.92}
		agent = {0.70, 0.78, 0.90}
		agent_core = {0.92, 0.95, 1.0}
		coral = {1.0, 0.38, 0.31}
		coral_core = {1.0, 0.66, 0.6}
	} else {
		you = {0.62, 0.53, 0.40}
		you_core = {0.47, 0.38, 0.26}
		agent = {0.40, 0.47, 0.58}
		agent_core = {0.24, 0.31, 0.43}
		coral = {0.9, 0.27, 0.21}
		coral_core = {0.7, 0.18, 0.13}
	}
	base := raster_saturate(raster_mix3(raster_mix3(you, agent, cool), coral, fail), 1.0)
	core := raster_saturate(
		raster_mix3(raster_mix3(you_core, agent_core, cool), coral_core, fail),
		1.0,
	)
	if dark {
		return RasterPalette {
			base = base,
			core = core,
			plate = {17.0 / 255.0, 17.0 / 255.0, 20.0 / 255.0, 0.94},
			rim = {1.0, 1.0, 1.0, 0.11},
			shadow = {0.0, 0.0, 0.0, 0.5},
		}
	}
	return RasterPalette {
		base = base,
		core = core,
		plate = {252.0 / 255.0, 251.0 / 255.0, 249.0 / 255.0, 0.96},
		rim = {0.0, 0.0, 0.0, 0.09},
		shadow = {20.0 / 255.0, 24.0 / 255.0, 40.0 / 255.0, 0.2},
	}
}

// `field_radius` with no island: the plate's corner radius is half its height,
// so its ends are pills. `u.field.x`, and the shader's `radius`.
raster_field_radius :: proc(pose_height: f32) -> f32 {
	return pose_height / 2.0
}

// The dot grid's per-dot terms, as the shader's inner loop computes them.
raster_dot_table :: proc(
	scene: RasterScene,
	pal: RasterPalette,
	half_size: RasterVec2,
	radius: f32,
) -> []RasterDot {
	columns := scene.columns
	rows := scene.rows
	pitch := scene.pitch
	center_offset := RasterVec2{(f32(columns) - 1.0) * 0.5, (f32(rows) - 1.0) * 0.5}
	dark := scene.dark
	unlit: f32 = dark ? 0.07 : 0.085
	full: f32 = dark ? 0.96 : 1.0
	curve: f32 = dark ? 0.9 : 0.75
	presence := 0.5 + 0.5 * max(scene.pose_live, scene.pose_cool)
	table := make([]RasterDot, columns * rows)
	for j in 0 ..< rows {
		for i in 0 ..< columns {
			center := RasterVec2{f32(i) - center_offset.x, f32(j) - center_offset.y}
			center.x *= pitch
			center.y *= pitch
			cap := -raster_rounded_box(center, half_size, radius)
			mask := raster_sstep(1.4, 4.6, cap)
			if mask <= 0 {
				table[j * columns + i] = RasterDot{center = center, mask = mask}
				continue
			}
			n := (j * columns + i) * 2
			power := max(f32(scene.dots[n]) / 65535.0, 1e-5)
			fresh := f32(scene.dots[n + 1]) / 65535.0
			row := abs(center.y) / half_size.y
			floor_alpha :=
				unlit * (1.0 - 0.7 * raster_sstep(0.35, 0.95, row)) * presence * scene.pose_grid
			alpha :=
				floor_alpha +
				(full - floor_alpha) * libc.powf(power, curve * raster_lerp(1.4, 1.0, fresh))
			dot_radius :=
				(0.25 + 0.17 * libc.powf(power, 0.6)) *
				raster_lerp(1.0, raster_lerp(0.82, 1.0, fresh), raster_sstep(0.03, 0.2, power)) *
				pitch
			core := clamp(power * power * 1.1, 0, 0.9) * raster_lerp(0.3, 1.0, fresh)
			table[j * columns + i] = RasterDot {
				center     = center,
				power      = power,
				mask       = mask,
				alpha      = alpha,
				dot_radius = dot_radius,
				color      = raster_mix3(pal.base, pal.core, core),
			}
		}
	}
	return table
}

// `dot_scope_fragment` at one pixel's `uv`, premultiplied.
raster_fragment :: proc(
	scene: RasterScene,
	pal: RasterPalette,
	table: []RasterDot,
	uv: RasterVec2,
) -> RasterRgba {
	resolution := RasterVec2{f32(scene.width), f32(scene.height)}
	position := RasterVec2{uv.x * resolution.x, uv.y * resolution.y}
	scale := max(scene.pose_scale, 0.001) * max(scene.layout_fit, 0.01)
	window_point := RasterVec2 {
		(position.x - resolution.x * 0.5) / scene.backing_scale,
		(position.y - resolution.y * 0.5) / scene.backing_scale,
	}
	point := RasterVec2 {
		(window_point.x - scene.layout_center_x) / scale,
		(window_point.y - scene.layout_center_y) / scale,
	}
	pixel := 1.0 / (scene.backing_scale * scale)
	softness := max(scene.pose_softness, 0)
	aa := pixel * 0.75 + softness * 0.72
	half_size := RasterVec2{scene.pose_width * 0.5, scene.pose_height * 0.5}
	radius := clamp(raster_field_radius(scene.pose_height), 0, scene.pose_height * 0.5)
	distance := raster_rounded_box(point, half_size, radius)
	dark := scene.dark
	result: RasterRgba

	// A soft drop shadow 3 pt below, as the prototype's 16 pt canvas blur.
	shadow_distance := raster_rounded_box(
		raster_sub(point, RasterVec2{0, 3}),
		half_size,
		radius,
	)
	raster_composite(
		&result,
		{pal.shadow.r, pal.shadow.g, pal.shadow.b},
		pal.shadow.a * (0.5 - 0.5 * libc.tanhf(shadow_distance / 6.5)),
	)

	shape := raster_coverage(distance, aa)
	raster_composite(&result, {pal.plate.r, pal.plate.g, pal.plate.b}, pal.plate.a * shape)
	raster_composite(
		&result,
		{pal.rim.r, pal.rim.g, pal.rim.b},
		pal.rim.a * raster_coverage(abs(distance) - 0.3, aa),
	)

	if scene.pose_glow > 0.01 {
		reach := 34.0 + 20.0 * scene.pose_glow
		t := clamp(raster_length(point) / reach, 0, 1)
		strength: f32 = (dark ? 0.3 : 0.19) * 0.55 * min(1.0, scene.pose_glow)
		middle := raster_mix3(pal.base, pal.core, 0.5)
		raster_composite(&result, raster_mix3(middle, pal.base, t), strength * (1 - t) * shape)
	}

	columns := scene.columns
	rows := scene.rows
	pitch := scene.pitch
	center_offset := RasterVec2{(f32(columns) - 1.0) * 0.5, (f32(rows) - 1.0) * 0.5}
	i0 := int(libc.roundf(point.x / pitch + center_offset.x))
	j0 := int(libc.roundf(point.y / pitch + center_offset.y))
	for dj in -1 ..= 1 {
		for di in -1 ..= 1 {
			i := i0 + di
			j := j0 + dj
			if i < 0 || j < 0 || i >= columns || j >= rows {
				continue
			}
			dot := table[j * columns + i]
			if dot.mask <= 0 {
				continue
			}
			from_center := raster_length(raster_sub(point, dot.center))
			if dot.power > 0.05 {
				raster_composite(
					&result,
					dot.color,
					0.132 * dot.power * dot.power * dot.mask * shape * raster_coverage(
						from_center - dot.dot_radius * 2.0,
						aa,
					),
				)
			}
			raster_composite(
				&result,
				dot.color,
				dot.alpha * dot.mask * shape * raster_coverage(from_center - dot.dot_radius, aa),
			)
		}
	}

	result = {
		result.r * scene.pose_opacity,
		result.g * scene.pose_opacity,
		result.b * scene.pose_opacity,
		result.a * scene.pose_opacity,
	}
	return raster_for_target(raster_dithered(result, position))
}

// A frame as premultiplied RGBA bytes, sRGB, `width * height * 4` of them. The
// caller owns the slice.
raster_render_rgba :: proc(scene: RasterScene) -> []u8 {
	needed := scene.columns * scene.rows * 2
	if len(scene.dots) < needed {
		panic("dot_scope_raster: RasterScene.dots is too short for the grid")
	}
	pal := raster_palette(scene.dark, scene.pose_cool, scene.pose_fail)
	half_size := RasterVec2{scene.pose_width * 0.5, scene.pose_height * 0.5}
	radius := clamp(raster_field_radius(scene.pose_height), 0, scene.pose_height * 0.5)
	table := raster_dot_table(scene, pal, half_size, radius)
	defer delete(table)

	width := f32(scene.width)
	height := f32(scene.height)
	out := make([]u8, scene.width * scene.height * 4)
	for y in 0 ..< scene.height {
		for x in 0 ..< scene.width {
			uv := RasterVec2{(f32(x) + 0.5) / width, (f32(y) + 0.5) / height}
			c := raster_fragment(scene, pal, table, uv)
			i := (y * scene.width + x) * 4
			out[i + 0] = raster_unorm(c.r)
			out[i + 1] = raster_unorm(c.g)
			out[i + 2] = raster_unorm(c.b)
			out[i + 3] = raster_unorm(c.a)
		}
	}
	return out
}

// A u8 render target's UNORM conversion: round to nearest, then clamp.
raster_unorm :: proc(v: f32) -> u8 {
	return u8(clamp(libc.roundf(v * 255.0), 0, 255))
}
