package phonon

// NeMo / HF `ParakeetFeatureExtractor` log-mel frontend, ported from
// phonon-rs-master/src/mel.rs:
//   preemphasis 0.97 -> 512-point centred STFT, symmetric 400-sample Hann
//   window, hop 160 -> power spectrum -> 128 Slaney mel bands ->
//   ln(x + 2^-24) -> per-band mean / unbiased-std normalisation over the
//   valid frames (the trailing padding frame is zeroed).
//
// 16 kHz, so 1056 ms of audio is ~106 frames; the FFT is a plain radix-2
// complex transform, which is far below the cost of the encoder.

import "core:fmt"
import "core:math"

MEL_N_MELS :: 128
MEL_N_FFT  :: 512
MEL_HOP    :: 160
MEL_WIN    :: 400
MEL_N_BINS :: MEL_N_FFT / 2 + 1
MEL_PREEMPH :: f32(0.97)
MEL_LOG_GUARD :: f32(5.960_464_5e-8) // 2^-24
MEL_EPS :: f32(1e-5)
MEL_SR :: 16000.0

Mel :: struct {
	filters: [MEL_N_MELS * MEL_N_BINS]f32,
	window:  [MEL_N_FFT]f32,
	twiddle: [MEL_N_FFT / 2]complex64, // exp(-2*pi*i*k/N), k < N/2
	rev:     [MEL_N_FFT]u16,           // bit-reversal permutation
	scratch: [MEL_N_FFT]complex64,
}

Mel_Features :: struct {
	data:   []f32, // frames * MEL_N_MELS, row-major
	frames: int,
	valid:  int,
}

// mel_init builds the window, mel filterbank and FFT tables once.
mel_init :: proc(m: ^Mel) {
	off := (MEL_N_FFT - MEL_WIN) / 2
	for n in 0 ..< MEL_WIN {
		// symmetric Hann: 0.5 - 0.5*cos(2*pi*n/(N-1))
		m.window[off + n] = f32(f64(0.5) - f64(0.5) * math.cos(2.0 * math.PI * f64(n) / f64(MEL_WIN - 1)))
	}
	bits := u32(9) // log2(512)
	for i in 0 ..< MEL_N_FFT {
		r: u32 = 0
		for b in 0 ..< bits {
			if (u32(i) >> b) & 1 != 0 { r |= u32(1) << (bits - 1 - b) }
		}
		m.rev[i] = u16(r)
	}
	for k in 0 ..< MEL_N_FFT / 2 {
		ang := -2.0 * math.PI * f64(k) / f64(MEL_N_FFT)
		m.twiddle[k] = complex(f32(math.cos(ang)), f32(math.sin(ang)))
	}
	mel_filterbank(&m.filters, MEL_SR, MEL_N_FFT, MEL_N_MELS)
}

// mel_filterbank reproduces librosa.filters.mel(sr, n_fft, n_mels,
// fmin=0, fmax=sr/2, htk=False, norm="slaney").
mel_filterbank :: proc(w: ^[MEL_N_MELS * MEL_N_BINS]f32, sr: f64, n_fft: int, n_mels: int) {
	hz_to_mel := proc(f: f64) -> f64 {
		F_SP :: 200.0 / 3.0
		MIN_LOG_HZ :: 1000.0
		min_log_mel := MIN_LOG_HZ / F_SP
		logstep := math.ln(6.4) / 27.0
		return f >= MIN_LOG_HZ ? min_log_mel + math.ln(f / MIN_LOG_HZ) / logstep : f / F_SP
	}
	mel_to_hz := proc(mv: f64) -> f64 {
		F_SP :: 200.0 / 3.0
		MIN_LOG_HZ :: 1000.0
		min_log_mel := MIN_LOG_HZ / F_SP
		logstep := math.ln(6.4) / 27.0
		return mv >= min_log_mel ? MIN_LOG_HZ * math.exp((mv - min_log_mel) * logstep) : mv * F_SP
	}

	n_bins := n_fft / 2 + 1
	lo := hz_to_mel(0.0)
	hi := hz_to_mel(sr / 2.0)
	mel_f := make([]f64, n_mels + 2, context.temp_allocator)
	for i in 0 ..< n_mels + 2 {
		mel_f[i] = mel_to_hz(lo + (hi - lo) * f64(i) / f64(n_mels + 1))
	}
	for i in 0 ..< n_mels {
		enorm := 2.0 / (mel_f[i + 2] - mel_f[i])
		d0 := mel_f[i + 1] - mel_f[i]
		d1 := mel_f[i + 2] - mel_f[i + 1]
		for k in 0 ..< n_bins {
			f := f64(k) * sr / f64(n_fft)
			lower := -(mel_f[i] - f) / d0
			upper := (mel_f[i + 2] - f) / d1
			v := math.max(math.min(lower, upper), 0.0)
			w[i * n_bins + k] = f32(v * enorm)
		}
	}
}

// mel_fft transforms m.scratch in place (forward, 512 points).
@(private)
mel_fft :: proc(m: ^Mel) {
	for i in 0 ..< MEL_N_FFT {
		j := int(m.rev[i])
		if i < j { m.scratch[i], m.scratch[j] = m.scratch[j], m.scratch[i] }
	}
	size := 2
	for size <= MEL_N_FFT {
		half := size / 2
		step := MEL_N_FFT / size
		for start in 0 ..< MEL_N_FFT {
			if start % size != 0 { continue }
			for k in 0 ..< half {
				t := m.twiddle[k * step] * m.scratch[start + k + half]
				u := m.scratch[start + k]
				m.scratch[start + k] = u + t
				m.scratch[start + k + half] = u - t
			}
		}
		size *= 2
	}
}

// mel_compute returns normalised features for mono 16 kHz samples. The caller
// owns the returned slice (allocator = context.allocator).
mel_compute :: proc(m: ^Mel, samples: []f32) -> Mel_Features {
	len_s := len(samples)
	padded := make([]f32, len_s + MEL_N_FFT, context.temp_allocator)
	half := MEL_N_FFT / 2
	for i in 0 ..< len_s {
		padded[half + i] = i == 0 ? samples[0] : samples[i] - MEL_PREEMPH * samples[i - 1]
	}

	frames := 1 + len_s / MEL_HOP
	valid := len_s / MEL_HOP
	data := make([]f32, frames * MEL_N_MELS)
	power := make([]f32, MEL_N_BINS, context.temp_allocator)

	for j in 0 ..< frames {
		start := j * MEL_HOP
		for k in 0 ..< MEL_N_FFT {
			m.scratch[k] = complex(padded[start + k] * m.window[k], 0)
		}
		mel_fft(m)
		for k in 0 ..< MEL_N_BINS {
			c := m.scratch[k]
			power[k] = real(c) * real(c) + imag(c) * imag(c)
		}
		row := data[j * MEL_N_MELS:][:MEL_N_MELS]
		for band in 0 ..< MEL_N_MELS {
			f := m.filters[band * MEL_N_BINS:][:MEL_N_BINS]
			s := f32(0)
			for k in 0 ..< MEL_N_BINS {
				s += f[k] * power[k]
			}
			row[band] = math.ln(s + MEL_LOG_GUARD)
		}
	}

	// per-band mean / unbiased std over the valid frames
	n := f32(max(valid, 1))
	mean := make([]f32, MEL_N_MELS, context.temp_allocator)
	var_ := make([]f32, MEL_N_MELS, context.temp_allocator)
	for t in 0 ..< valid {
		row := data[t * MEL_N_MELS:][:MEL_N_MELS]
		for band in 0 ..< MEL_N_MELS { mean[band] += row[band] }
	}
	for band in 0 ..< MEL_N_MELS { mean[band] /= n }
	for t in 0 ..< valid {
		row := data[t * MEL_N_MELS:][:MEL_N_MELS]
		for band in 0 ..< MEL_N_MELS {
			d := row[band] - mean[band]
			var_[band] += d * d
		}
	}
	denom := f32(max(valid, 2) - 1)
	inv := make([]f32, MEL_N_MELS, context.temp_allocator)
	for band in 0 ..< MEL_N_MELS { inv[band] = 1.0 / (math.sqrt(var_[band] / denom) + MEL_EPS) }
	for t in 0 ..< frames {
		row := data[t * MEL_N_MELS:][:MEL_N_MELS]
		for band in 0 ..< MEL_N_MELS {
			row[band] = t < valid ? (row[band] - mean[band]) * inv[band] : 0.0
		}
	}
	return Mel_Features{data = data, frames = frames, valid = valid}
}

