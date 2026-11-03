package phonon

// GGUF v3 reader for the Phonon GGUF (parakeet TDT / Fermion).
//
// The file is read once into memory and tensors are zero-copy views into that
// buffer, so quantization is decoded only when a kernel consumes it. Tensor
// payloads stay in their on-disk type (Q8_0 blocks for most weights).
//
// Reference: phonon-rs-master/fermion.rs parses the same file; the tensor names
// and layout come from there.

import "core:fmt"
import "core:mem"
import "core:os"
import "core:strings"

GGUF_MAGIC :: u32(0x4655_4747) // "GGUF" as little-endian u32

Gguf_Type :: enum u32 {
	UINT8   = 0,
	INT8    = 1,
	UINT16  = 2,
	INT16   = 3,
	UINT32  = 4,
	INT32   = 5,
	FLOAT32 = 6,
	BOOL    = 7,
	STRING  = 8,
	ARRAY   = 9,
	UINT64  = 10,
	INT64   = 11,
	FLOAT64 = 12,
}

// ggml tensor type ids as written by the converter.
GGML_Type :: enum u32 {
	F32   = 0,
	F16   = 1,
	Q4_0  = 2,
	Q4_1  = 3,
	Q5_0  = 6,
	Q5_1  = 7,
	Q8_0  = 8,
	Q8_1  = 9,
	Q2_K  = 10,
	Q3_K  = 11,
	Q4_K  = 12,
	Q5_K  = 13,
	Q6_K  = 14,
	Q8_K  = 15,
	I8    = 24,
	I16   = 25,
	I32   = 26,
	I64   = 27,
	BF16  = 28,
	F64   = 34,
}

Gguf_Value :: union {
	bool,
	i64,
	f32,
	f64,
	string,
	[]i64,
	[]string,
}

Gguf_Tensor :: struct {
	name:   string,
	dims:   [4]u64,
	n_dims: u32,
	type:   GGML_Type,
	offset: u64,   // byte offset into the tensor data region
	bytes:  []u8,  // resolved view, valid while Gguf.data is alive
	// scales holds the block scales of a Q8_0 tensor (one f32 per 32 values,
	// row-major) decoded once at load: the dot kernel reads four of them with
	// one vector load instead of decoding four f16 splats per block group.
	scales: []f32,
}

Gguf :: struct {
	data:     []u8, // whole file
	tensors:  map[string]Gguf_Tensor,
	kv:       map[string]Gguf_Value,
	order:    []string, // tensor names in file order
	alignment: u64,
}

// Elements per block and bytes per block; 1/1 for plain types.
ggml_type_layout :: proc(t: GGML_Type) -> (elems: u64, bytes: u64, ok: bool) {
	switch t {
	case .F32, .I32:        return 1, 4, true
	case .F64, .I64:        return 1, 8, true
	case .F16, .BF16, .I16: return 1, 2, true
	case .I8:               return 1, 1, true
	case .Q4_0:         return 32, 18, true
	case .Q4_1:         return 32, 20, true
	case .Q5_0:         return 32, 22, true
	case .Q5_1:         return 32, 24, true
	case .Q8_0:         return 32, 34, true
	case .Q8_1:         return 32, 36, true
	case .Q2_K:         return 256, 84, true
	case .Q3_K:         return 256, 110, true
	case .Q4_K:         return 256, 144, true
	case .Q5_K:         return 256, 176, true
	case .Q6_K:         return 256, 210, true
	case .Q8_K:         return 256, 292, true
	}
	return 0, 0, false
}

gguf_tensor_bytes :: proc(t: GGML_Type, n_elements: u64) -> (u64, bool) {
	elems, bytes, ok := ggml_type_layout(t)
	if !ok { return 0, false }
	if n_elements % elems != 0 { return 0, false }
	return (n_elements / elems) * bytes, true
}

@(private)
Reader :: struct {
	data: []u8,
	pos:  int,
}

@(private)
read_bytes :: proc(r: ^Reader, n: int) -> []u8 {
	if r.pos + n > len(r.data) { return nil }
	b := r.data[r.pos:r.pos + n]
	r.pos += n
	return b
}

@(private)
read_pod :: proc(r: ^Reader, $T: typeid) -> T {
	v: T
	b := read_bytes(r, size_of(T))
	if b == nil { return v }
	return (cast(^T)(raw_data(b)))^
}

@(private)
read_string :: proc(r: ^Reader, alloc: mem.Allocator) -> string {
	n := read_pod(r, u64)
	b := read_bytes(r, int(n))
	if b == nil { return "" }
	return strings.clone(string(b), alloc)
}

@(private)
skip_value :: proc(r: ^Reader, t: Gguf_Type, alloc: mem.Allocator) {
	switch t {
	case .UINT8, .INT8, .BOOL:          _ = read_bytes(r, 1)
	case .UINT16, .INT16:               _ = read_bytes(r, 2)
	case .UINT32, .INT32, .FLOAT32:     _ = read_bytes(r, 4)
	case .UINT64, .INT64, .FLOAT64:     _ = read_bytes(r, 8)
	case .STRING:                       _ = read_string(r, alloc)
	case .ARRAY:
		et := read_pod(r, Gguf_Type)
		n := read_pod(r, u64)
		for _ in 0 ..< n { skip_value(r, et, alloc) }
	}
}

@(private)
read_value :: proc(r: ^Reader, t: Gguf_Type, alloc: mem.Allocator) -> Gguf_Value {
	switch t {
	case .UINT8:   return i64(read_pod(r, u8))
	case .INT8:    return i64(read_pod(r, i8))
	case .BOOL:    return read_pod(r, u8) != 0
	case .UINT16:  return i64(read_pod(r, u16))
	case .INT16:   return i64(read_pod(r, i16))
	case .UINT32:  return i64(read_pod(r, u32))
	case .INT32:   return i64(read_pod(r, i32))
	case .FLOAT32: return read_pod(r, f32)
	case .UINT64:  return i64(read_pod(r, u64))
	case .INT64:   return read_pod(r, i64)
	case .FLOAT64: return read_pod(r, f64)
	case .STRING:  return read_string(r, alloc)
	case .ARRAY:
		et := read_pod(r, Gguf_Type)
		n := read_pod(r, u64)
		switch et {
		case .STRING:
			out := make([]string, int(n), allocator = alloc)
			for i in 0 ..< int(n) { out[i] = read_string(r, alloc) }
			return out
		case .UINT8, .INT8, .BOOL, .UINT16, .INT16, .UINT32, .INT32, .UINT64, .INT64:
			out := make([]i64, int(n), allocator = alloc)
			for i in 0 ..< int(n) {
				v := read_value(r, et, alloc)
				out[i] = v.(i64)
			}
			return out
		case .FLOAT32, .FLOAT64, .ARRAY:
			// float arrays are not needed for this model; skip them
			for _ in 0 ..< n { skip_value(r, et, alloc) }
			return nil
		}
	}
	return nil
}

// gguf_load reads and parses the file. Coalesce sparse metadata reads: the
// whole file is mapped into one buffer, which is what the encoder streams
// anyway.
gguf_load :: proc(path: string, alloc := context.allocator) -> (g: Gguf, err: string) {
	data, rerr := os.read_entire_file_from_path(path, alloc)
	if rerr != nil { return {}, fmt.tprintf("gguf: cannot read %s: %v", path, rerr) }
	g.data = data
	g.alignment = 32
	g.tensors = make(map[string]Gguf_Tensor, 1024)
	g.kv = make(map[string]Gguf_Value, 64)

	r := Reader{data = data}
	magic := read_pod(&r, u32)
	if magic != GGUF_MAGIC { return {}, "gguf: bad magic" }
	version := read_pod(&r, u32)
	if version != 3 { return {}, fmt.tprintf("gguf: unsupported version %d", version) }
	n_tensors := read_pod(&r, u64)
	n_kv := read_pod(&r, u64)

	for _ in 0 ..< n_kv {
		key := read_string(&r, alloc)
		vt := read_pod(&r, Gguf_Type)
		val := read_value(&r, vt, alloc)
		if key != "" { g.kv[key] = val }
	}
	if a, ok := g.kv["general.alignment"]; ok {
		if ai, iok := a.(i64); iok { g.alignment = u64(ai) }
	}
	if g.alignment == 0 { g.alignment = 32 }

	infos := make([]Gguf_Tensor, int(n_tensors), context.temp_allocator)
	g.order = make([]string, int(n_tensors), allocator = alloc)
	for i in 0 ..< int(n_tensors) {
		name := read_string(&r, alloc)
		n_dims := read_pod(&r, u32)
		if n_dims > 4 { return {}, fmt.tprintf("gguf: tensor %s has %d dims", name, n_dims) }
		t := Gguf_Tensor{name = name, n_dims = n_dims}
		for d in 0 ..< int(n_dims) { t.dims[d] = read_pod(&r, u64) }
		t.type = read_pod(&r, GGML_Type)
		t.offset = read_pod(&r, u64)
		infos[i] = t
		g.order[i] = name
	}

	// The tensor data region starts after the info block, aligned up.
	data_start := (u64(r.pos) + g.alignment - 1) / g.alignment * g.alignment
	if data_start > u64(len(data)) { return {}, "gguf: truncated tensor data" }
	for t in infos {
		n_elements := u64(1)
		for d in 0 ..< int(t.n_dims) { n_elements *= t.dims[d] }
		nbytes, ok := gguf_tensor_bytes(t.type, n_elements)
		if !ok { return {}, fmt.tprintf("gguf: tensor %s type %d unsupported", t.name, t.type) }
		start := data_start + t.offset
		if start + nbytes > u64(len(data)) {
			return {}, fmt.tprintf("gguf: tensor %s out of bounds", t.name)
		}
		tt := t
		tt.bytes = data[int(start):int(start + nbytes)]
		if tt.type == .Q8_0 {
			blocks := int(n_elements / Q8_0_QUANT_SIZEOF)
			tt.scales = make([]f32, blocks, alloc)
			for b in 0 ..< blocks {
				tt.scales[b] = half_to_float((cast(^u16)&tt.bytes[b * Q8_0_BLOCK_SIZEOF])^)
			}
		}
		g.tensors[t.name] = tt
	}
	return g, ""
}

gguf_destroy :: proc(g: ^Gguf) {
	for _, t in g.tensors {
		if t.scales != nil { delete(t.scales) }
	}
	if g.data != nil { delete(g.data) }
	delete(g.tensors)
	delete(g.kv)
	delete(g.order)
}

// gguf_int returns an integer metadata entry, or def.
gguf_int :: proc(g: ^Gguf, key: string, def: i64) -> i64 {
	if v, ok := g.kv[key]; ok {
		if i, iok := v.(i64); iok { return i }
		if f, fok := v.(f32); fok { return i64(f) }
	}
	return def
}

gguf_str :: proc(g: ^Gguf, key: string) -> string {
	if v, ok := g.kv[key]; ok {
		if s, sok := v.(string); sok { return s }
	}
	return ""
}

gguf_tensor :: proc(g: ^Gguf, name: string) -> (Gguf_Tensor, bool) {
	t, ok := g.tensors[name]
	return t, ok
}

gguf_elements :: proc(t: Gguf_Tensor) -> u64 {
	n := u64(1)
	for d in 0 ..< int(t.n_dims) { n *= t.dims[d] }
	return n
}

gguf_row_size :: proc(t: Gguf_Tensor) -> int {
	// Bytes for one row (the slowest-varying axis is dims[n_dims-1]).
	if t.n_dims == 0 { return 0 }
	elements_per_row := t.dims[0]
	nbytes, _ := gguf_tensor_bytes(t.type, elements_per_row)
	return int(nbytes)
}

// gguf_report prints an inventory of the file (used by `hex gguf`).
gguf_report :: proc(path: string, list_tensors: bool) -> bool {
	g, err := gguf_load(path)
	if err != "" {
		fmt.eprintln(err)
		return false
	}
	defer gguf_destroy(&g)

	quant := map[GGML_Type]int{}
	total_elems := u64(0)
	total_bytes := u64(0)
	for name in g.order {
		t := g.tensors[name]
		quant[t.type] += 1
		total_elems += gguf_elements(t)
		total_bytes += u64(len(t.bytes))
	}

	fmt.printf("file      %s\n", path)
	fmt.printf("bytes     %d\n", len(g.data))
	fmt.printf("arch      %s (%s, %s)\n", gguf_str(&g, "general.architecture"),
		gguf_str(&g, "general.name"), gguf_str(&g, "general.size_label"))
	fmt.printf("layers    %d  d_model %d  heads %d  ff %d\n",
		gguf_int(&g, "stt.parakeet.encoder.n_layers", 0),
		gguf_int(&g, "stt.parakeet.encoder.d_model", 0),
		gguf_int(&g, "stt.parakeet.encoder.n_heads", 0),
		gguf_int(&g, "stt.parakeet.encoder.d_ff", 0))
	fmt.printf("tensors   %d  elements %d  payload %d bytes\n", len(g.order), total_elems, total_bytes)
	for t, n in quant {
		fmt.printf("  type %-5v %d tensors\n", t, n)
	}
	if list_tensors {
		for name in g.order {
			t := g.tensors[name]
			fmt.printf("%-44s %-5v dims=%v bytes=%d\n", name, t.type, t.dims[:t.n_dims], len(t.bytes))
		}
	}
	return true
}
