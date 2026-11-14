package main

// Live bridge from the real Dot scope simulation to the HUD DIB. The window,
// hotkeys, capture and transcription worker are unchanged; only the pixels
// come from the same rasteriser that produces the deterministic frames.
hud_render_scope :: proc(frame: DotFrame) {
	if hud.w <= 0 || hud.h <= 0 || len(frame.dots) < frame.columns * frame.rows * 2 {
		return
	}
	p := frame.pose
	scene := RasterScene{
		width = hud.w,
		height = hud.h,
		backing_scale = hud.scale,
		dark = true,
		pitch = PITCH,
		columns = frame.columns,
		rows = frame.rows,
		dots = frame.dots,
		layout_center_x = 0,
		layout_center_y = -3,
		layout_fit = 1,
		pose_width = p.width,
		pose_height = p.height,
		pose_opacity = p.opacity,
		pose_scale = p.scale,
		pose_softness = p.softness,
		pose_cool = p.cool,
		pose_fail = p.fail,
		pose_glow = p.glow,
		pose_grid = p.grid,
		pose_live = p.live,
		pose_reveal = p.reveal,
		pose_level = p.level,
		pose_clock = p.clock,
		pose_done = p.done,
	}
	rgba := raster_render_rgba(scene)
	defer delete(rgba)
	if len(rgba) != hud.w * hud.h * 4 {
		return
	}
	for i in 0 ..< hud.w * hud.h {
		r := u32(rgba[i * 4 + 0])
		g := u32(rgba[i * 4 + 1])
		b := u32(rgba[i * 4 + 2])
		a := u32(rgba[i * 4 + 3])
		hud.pixels[i] = b | (g << 8) | (r << 16) | (a << 24)
	}
}
