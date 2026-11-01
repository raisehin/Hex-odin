package phonon

// conv2d dispatches on the conv shape (im2col matmul, depthwise, 1x1 matmul).
// These check each path against the naive definition over the shapes the
// subsampling stack actually asks for, including the GGUF weight layout
// [kx][ky][ic][oc], which the matmul paths have to see as [oc][cols].

import "core:fmt"
import "core:testing"

naive_conv2d :: proc(
	x: []f32, ics, it, ifr: int,
	w: []f32, kx, ky, ocs, groups: int,
	bias: []f32,
	s, p: int,
) -> (out: []f32, ot, ofr: int) {
	ot = (it + 2 * p - ky) / s + 1
	ofr = (ifr + 2 * p - kx) / s + 1
	out = make([]f32, ocs * ot * ofr)
	in_per_group := ics / groups
	out_per_group := ocs / groups
	for o in 0 ..< ocs {
		grp := o / out_per_group
		b := bias == nil ? 0 : bias[o]
		plane := out[o * ot * ofr:]
		for ty in 0 ..< ot {
			for fx in 0 ..< ofr {
				acc := b
				for ky_i in 0 ..< ky {
					iy := ty * s - p + ky_i
					if iy < 0 || iy >= it { continue }
					for kx_i in 0 ..< kx {
						ix := fx * s - p + kx_i
						if ix < 0 || ix >= ifr { continue }
						for ic in 0 ..< in_per_group {
							wv := w[(((o * in_per_group + ic) * ky + ky_i) * kx) + kx_i]
							acc += wv * x[(grp * in_per_group + ic) * it * ifr + iy * ifr + ix]
						}
					}
				}
				plane[ty * ofr + fx] = acc
			}
		}
	}
	return
}

check_shape :: proc(t: ^testing.T, ics, it, ifr, ocs, kx, ky, groups, s, p: int) {
	n_in := ics * it * ifr
	n_w := kx * ky * (ics / groups) * ocs
	x := make([]f32, n_in, context.temp_allocator)
	w := make([]f32, n_w, context.temp_allocator)
	bias := make([]f32, ocs, context.temp_allocator)
	for i in 0 ..< n_in { x[i] = f32((i * 37 % 101) - 50) / 41 }
	for i in 0 ..< n_w { w[i] = f32((i * 53 % 97) - 48) / 23 }
	for i in 0 ..< ocs { bias[i] = f32(i % 7) / 3 }

	want, wt, wf := naive_conv2d(x, ics, it, ifr, w, kx, ky, ocs, groups, bias, s, p)
	got, gt, gf := conv2d(x, ics, it, ifr, w, kx, ky, ocs, groups, bias, s, p, nil, 1)
	testing.expect(t, wt == gt && wf == gf, "shape mismatch")
	if wt != gt || wf != gf {
		delete(want)
		delete(got)
		return
	}
	worst: f32 = 0
	wi := 0
	for v, i in want {
		d := abs(v - got[i])
		if d > worst { worst = d; wi = i }
	}
	fmt.printfln("shape ic=%d it=%d ifr=%d oc=%d g=%d k=%dx%d s=%d p=%d -> [%d][%d][%d] worst %.3e at %d",
		ics, it, ifr, ocs, groups, kx, ky, s, p, gt, gf, ocs, worst, wi)
	testing.expectf(t, worst <= 1e-3 * max(1, abs(want[wi])), "worst diff %v", worst)
	delete(want)
	delete(got)
}

@(test)
conv2d_matches_naive :: proc(t: ^testing.T) {
	// conv.0: 1 -> 256, 3x3 stride 2 pad 1
	check_shape(t, 1, 112, 80, 256, 3, 3, 1, 2, 1)
	// conv.2 / conv.5: depthwise 3x3 stride 2 pad 1
	check_shape(t, 256, 56, 40, 256, 3, 3, 256, 2, 1)
	check_shape(t, 256, 28, 20, 256, 3, 3, 256, 2, 1)
	// conv.3 / conv.6: pointwise 256 -> 256
	check_shape(t, 256, 28, 20, 256, 1, 1, 1, 1, 0)
	check_shape(t, 256, 14, 10, 256, 1, 1, 1, 1, 0)
}
