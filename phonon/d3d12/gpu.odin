// D3D12 compute plumbing for the phonon encoder.
//
// Scope: plain committed buffers, root SRV/UAV descriptors and 32-bit root
// constants. No heaps, no descriptor tables, no swap chain -- the encoder is a
// sequence of GEMM-like dispatches, and this is the smallest surface that can
// run them. Everything here is Windows-only and links the in-box d3d12.lib, so
// it needs no SDK beyond the Windows one Odin already targets.
//
// The bindings in vendor:directx/d3d12 strip the D3D12 prefix from COM types:
// IDevice, ICommandQueue, IGraphicsCommandList, IResource, not ID3D12Device.

package phonon_d3d12

import "core:fmt"
import "core:mem"
import "core:sys/windows"
import "vendor:directx/d3d12"

STATE_UNORDERED_ACCESS :: d3d12.RESOURCE_STATES{.UNORDERED_ACCESS}
STATE_COPY_SOURCE      :: d3d12.RESOURCE_STATES{.COPY_SOURCE}
STATE_COPY_DEST        :: d3d12.RESOURCE_STATES{.COPY_DEST}
STATE_COMMON           :: d3d12.RESOURCE_STATE_COMMON

Gpu :: struct {
	device:      ^d3d12.IDevice,
	queue:       ^d3d12.ICommandQueue,
	alloc:       ^d3d12.ICommandAllocator,
	list:        ^d3d12.IGraphicsCommandList,
	fence:       ^d3d12.IFence,
	event:       windows.HANDLE,
	root_sig:    ^d3d12.IRootSignature,
	pso:         ^d3d12.IPipelineState,
	fence_value: u64,
	signaled:    u64, // last fence value flushed to the queue
}

Buffer :: struct {
	res:    ^d3d12.IResource,
	size:   u64,
	heap:   d3d12.HEAP_TYPE,
	mapped: rawptr,
}

hr_ok :: proc(hr: d3d12.HRESULT) -> bool { return hr >= 0 }

hresult_msg :: proc(hr: d3d12.HRESULT) -> string {
	return fmt.tprintf("HRESULT 0x%08x", u32(hr))
}

// Root parameter layout, shared by every shader in this package:
//   0  b0  8 x u32 root constants (GEMM shape, flags)
//   1  t0  SRV weight quants
//   2  t1  SRV weight scales
//   3  t2  SRV activation quants
//   4  t3  SRV activation scales
//   5  t4  SRV bias
//   6  u0  UAV output
ROOT_CONSTANTS_INDEX  :: 0
ROOT_WEIGHTS_Q_INDEX  :: 1
ROOT_WEIGHTS_D_INDEX  :: 2
ROOT_ACT_Q_INDEX      :: 3
ROOT_ACT_D_INDEX      :: 4
ROOT_BIAS_INDEX       :: 5
ROOT_OUT_INDEX        :: 6
ROOT_SRV_COUNT        :: 5
ROOT_CONSTANT_COUNT   :: 8

gpu_init :: proc(shader: []u8) -> (g: ^Gpu, err: string) {
	g = new(Gpu)

	dev_raw: rawptr
	hr := d3d12.CreateDevice(nil, ._12_0, d3d12.IDevice_UUID, &dev_raw)
	if !hr_ok(hr) { free(g); return nil, fmt.tprintf("D3D12CreateDevice: %v", hresult_msg(hr)) }
	g.device = cast(^d3d12.IDevice)dev_raw

	qdesc := d3d12.COMMAND_QUEUE_DESC {
		Type     = .COMPUTE,
		Priority = i32(d3d12.COMMAND_QUEUE_PRIORITY.NORMAL),
		Flags    = {},
		NodeMask = 0,
	}
	q_raw: rawptr
	hr = g.device->CreateCommandQueue(&qdesc, d3d12.ICommandQueue_UUID, &q_raw)
	if !hr_ok(hr) { gpu_destroy(g); return nil, fmt.tprintf("CreateCommandQueue: %v", hresult_msg(hr)) }
	g.queue = cast(^d3d12.ICommandQueue)q_raw

	if rerr := gpu_create_root_signature(g); rerr != "" { gpu_destroy(g); return nil, rerr }
	if perr := gpu_create_pipeline(g, shader); perr != "" { gpu_destroy(g); return nil, perr }

	a_raw: rawptr
	hr = g.device->CreateCommandAllocator(.COMPUTE, d3d12.ICommandAllocator_UUID, &a_raw)
	if !hr_ok(hr) { gpu_destroy(g); return nil, fmt.tprintf("CreateCommandAllocator: %v", hresult_msg(hr)) }
	g.alloc = cast(^d3d12.ICommandAllocator)a_raw

	l_raw: rawptr
	hr = g.device->CreateCommandList(0, .COMPUTE, g.alloc, nil, d3d12.IGraphicsCommandList_UUID, &l_raw)
	if !hr_ok(hr) { gpu_destroy(g); return nil, fmt.tprintf("CreateCommandList: %v", hresult_msg(hr)) }
	g.list = cast(^d3d12.IGraphicsCommandList)l_raw
	g.list->Close()

	f_raw: rawptr
	hr = g.device->CreateFence(0, {}, d3d12.IFence_UUID, &f_raw)
	if !hr_ok(hr) { gpu_destroy(g); return nil, fmt.tprintf("CreateFence: %v", hresult_msg(hr)) }
	g.fence = cast(^d3d12.IFence)f_raw

	g.event = windows.CreateEventW(nil, false, false, nil)
	if g.event == nil { gpu_destroy(g); return nil, "CreateEventW failed" }

	return g, ""
}

@(private = "file")
gpu_create_root_signature :: proc(g: ^Gpu) -> string {
	params := [ROOT_SRV_COUNT + 2]d3d12.ROOT_PARAMETER{}
	params[0].ParameterType = ._32BIT_CONSTANTS
	params[0].Constants = d3d12.ROOT_CONSTANTS{ShaderRegister = 0, RegisterSpace = 0, Num32BitValues = ROOT_CONSTANT_COUNT}
	params[0].ShaderVisibility = .ALL

	for i in 1 ..= ROOT_SRV_COUNT {
		params[i].ParameterType = .SRV
		params[i].Descriptor = d3d12.ROOT_DESCRIPTOR{ShaderRegister = u32(i - 1), RegisterSpace = 0}
		params[i].ShaderVisibility = .ALL
	}
	params[ROOT_SRV_COUNT + 1].ParameterType = .UAV
	params[ROOT_SRV_COUNT + 1].Descriptor = d3d12.ROOT_DESCRIPTOR{ShaderRegister = 0, RegisterSpace = 0}
	params[ROOT_SRV_COUNT + 1].ShaderVisibility = .ALL

	desc := d3d12.ROOT_SIGNATURE_DESC {
		NumParameters     = u32(len(params)),
		pParameters       = &params[0],
		NumStaticSamplers = 0,
		pStaticSamplers   = nil,
		Flags             = {},
	}

	blob: ^d3d12.IBlob
	err_blob: ^d3d12.IBlob
	hr := d3d12.SerializeRootSignature(&desc, ._1, &blob, &err_blob)
	if !hr_ok(hr) {
		msg := "SerializeRootSignature failed"
		if err_blob != nil {
			ptr := err_blob->GetBufferPointer()
			n := err_blob->GetBufferSize()
			msg = fmt.tprintf("SerializeRootSignature: %s", string(cast(cstring)ptr))
			err_blob->Release()
		}
		return msg
	}
	defer blob->Release()

	rs_raw: rawptr
	hr = g.device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), d3d12.IRootSignature_UUID, &rs_raw)
	if !hr_ok(hr) { return fmt.tprintf("CreateRootSignature: %v", hresult_msg(hr)) }
	g.root_sig = cast(^d3d12.IRootSignature)rs_raw
	return ""
}

@(private = "file")
gpu_create_pipeline :: proc(g: ^Gpu, shader: []u8) -> string {
	desc := d3d12.COMPUTE_PIPELINE_STATE_DESC {
		pRootSignature = g.root_sig,
		CS             = d3d12.SHADER_BYTECODE{pShaderBytecode = raw_data(shader), BytecodeLength = d3d12.SIZE_T(len(shader))},
		NodeMask       = 0,
		CachedPSO      = {},
		Flags          = {},
	}
	pso_raw: rawptr
	hr := g.device->CreateComputePipelineState(&desc, d3d12.IPipelineState_UUID, &pso_raw)
	if !hr_ok(hr) { return fmt.tprintf("CreateComputePipelineState: %v", hresult_msg(hr)) }
	g.pso = cast(^d3d12.IPipelineState)pso_raw
	return ""
}

gpu_destroy :: proc(g: ^Gpu) {
	if g == nil { return }
	if g.event != nil { windows.CloseHandle(g.event) }
	if g.pso != nil { g.pso->Release() }
	if g.root_sig != nil { g.root_sig->Release() }
	if g.fence != nil { g.fence->Release() }
	if g.list != nil { g.list->Release() }
	if g.alloc != nil { g.alloc->Release() }
	if g.queue != nil { g.queue->Release() }
	if g.device != nil { g.device->Release() }
	free(g)
}

// gpu_flush enqueues a fence signal so the submitted work starts on the
// GPU now, not at the next wait: some drivers defer the doorbell until a
// sync point, which would serialize a GPU+CPU row-split. Pair with gpu_join.
gpu_flush :: proc(g: ^Gpu) -> string {
	g.fence_value += 1
	hr := g.queue->Signal(g.fence, g.fence_value)
	if !hr_ok(hr) { return fmt.tprintf("Signal: %v", hresult_msg(hr)) }
	g.signaled = g.fence_value
	return ""
}

// gpu_join blocks until the last flushed submission has completed.
gpu_join :: proc(g: ^Gpu) -> string {
	hr := g.fence->SetEventOnCompletion(g.signaled, g.event)
	if !hr_ok(hr) { return fmt.tprintf("SetEventOnCompletion: %v", hresult_msg(hr)) }
	rc := windows.WaitForSingleObject(g.event, 0xFFFFFFFF)
	if rc != 0 { return fmt.tprintf("WaitForSingleObject returned %v", rc) }
	return ""
}

// gpu_wait blocks until every command submitted so far has completed.
gpu_wait :: proc(g: ^Gpu) -> string {
	if err := gpu_flush(g); err != "" { return err }
	return gpu_join(g)
}

@(private = "file")
heap_props :: proc(heap: d3d12.HEAP_TYPE) -> d3d12.HEAP_PROPERTIES {
	// This is what CD3DX12_HEAP_PROPERTIES(heap_type) fills in. DEFAULT with
	// CPU_PAGE_PROPERTY_NOT_AVAILABLE is rejected with E_INVALIDARG: the CPU
	// page property is only meaningful for CUSTOM heaps.
	return d3d12.HEAP_PROPERTIES {
		Type                 = heap,
		CPUPageProperty      = .UNKNOWN,
		MemoryPoolPreference = .UNKNOWN,
		CreationNodeMask     = 1,
		VisibleNodeMask      = 1,
	}
}

buffer_create :: proc(g: ^Gpu, size: u64, heap: d3d12.HEAP_TYPE, state: d3d12.RESOURCE_STATES, flags := d3d12.RESOURCE_FLAGS{}) -> (b: ^Buffer, err: string) {
	sz := max(size, 4)
	b = new(Buffer)
	b.size = sz
	b.heap = heap

	props := heap_props(heap)
	desc := d3d12.RESOURCE_DESC {
		Dimension        = .BUFFER,
		Alignment        = 0,
		Width            = sz,
		Height           = 1,
		DepthOrArraySize = 1,
		MipLevels        = 1,
		Format           = .UNKNOWN,
		SampleDesc       = {Count = 1, Quality = 0},
		Layout           = .ROW_MAJOR,
		Flags            = flags,
	}
	raw: rawptr
	hr := g.device->CreateCommittedResource(&props, {}, &desc, state, nil, d3d12.IResource_UUID, &raw)
	if !hr_ok(hr) { free(b); return nil, fmt.tprintf("CreateCommittedResource(%v, %d bytes): %v", heap, sz, hresult_msg(hr)) }
	b.res = cast(^d3d12.IResource)raw

	if heap != .DEFAULT {
		hr = b.res->Map(0, nil, &b.mapped)
		if !hr_ok(hr) { b.res->Release(); free(b); return nil, fmt.tprintf("Map: %v", hresult_msg(hr)) }
	}
	return b, ""
}

buffer_destroy :: proc(b: ^Buffer) {
	if b == nil { return }
	if b.mapped != nil { b.res->Unmap(0, nil) }
	if b.res != nil { b.res->Release() }
	free(b)
}

buffer_write :: proc(b: ^Buffer, data: []u8) -> string {
	if b.mapped == nil { return "buffer_write on a non-mapped buffer" }
	if u64(len(data)) > b.size { return fmt.tprintf("buffer_write overflow: %d > %d", len(data), b.size) }
	mem.copy(b.mapped, raw_data(data), len(data))
	return ""
}

buffer_read :: proc(b: ^Buffer, dst: []u8) -> string {
	if b.mapped == nil { return "buffer_read on a non-mapped buffer" }
	if u64(len(dst)) > b.size { return fmt.tprintf("buffer_read overflow: %d > %d", len(dst), b.size) }
	mem.copy(raw_data(dst), b.mapped, len(dst))
	return ""
}

transition :: proc(list: ^d3d12.IGraphicsCommandList, res: ^d3d12.IResource, before, after: d3d12.RESOURCE_STATES) {
	barrier := d3d12.RESOURCE_BARRIER {
		Type  = .TRANSITION,
		Flags = {},
		Transition = d3d12.RESOURCE_TRANSITION_BARRIER {
			pResource   = res,
			Subresource = 0,
			StateBefore = before,
			StateAfter  = after,
		},
	}
	list->ResourceBarrier(1, &barrier)
}

// dispatch_launch records and submits one compute dispatch with this
// package's root layout, without waiting: the caller overlaps CPU work with
// the GPU, then calls dispatch_finish. `constants` is the b0 payload
// (ROOT_CONSTANT_COUNT u32 values).
dispatch_launch :: proc(
	g: ^Gpu,
	groups: [3]u32,
	constants: ^[ROOT_CONSTANT_COUNT]u32,
	srvs: [ROOT_SRV_COUNT]^Buffer,
	uav: ^Buffer,
	uav_before: d3d12.RESOURCE_STATES,
	dst: ^Buffer,
	readback: ^Buffer,
	out_bytes: int,
) -> string {
	if hr := g.alloc->Reset(); !hr_ok(hr) { return fmt.tprintf("alloc Reset: %v", hresult_msg(hr)) }
	if hr := g.list->Reset(g.alloc, g.pso); !hr_ok(hr) { return fmt.tprintf("list Reset: %v", hresult_msg(hr)) }

	transition(g.list, uav.res, uav_before, STATE_UNORDERED_ACCESS)
	g.list->SetComputeRootSignature(g.root_sig)
	g.list->SetPipelineState(g.pso)
	g.list->SetComputeRoot32BitConstants(ROOT_CONSTANTS_INDEX, ROOT_CONSTANT_COUNT, constants, 0)
	for b, i in srvs {
		if b != nil {
			g.list->SetComputeRootShaderResourceView(u32(ROOT_WEIGHTS_Q_INDEX + i), b.res->GetGPUVirtualAddress())
		}
	}
	g.list->SetComputeRootUnorderedAccessView(ROOT_OUT_INDEX, uav.res->GetGPUVirtualAddress())
	g.list->Dispatch(groups[0], groups[1], groups[2])

	transition(g.list, uav.res, STATE_UNORDERED_ACCESS, STATE_COPY_SOURCE)
	g.list->CopyBufferRegion(readback.res, 0, dst.res, 0, u64(out_bytes))

	if hr := g.list->Close(); !hr_ok(hr) { return fmt.tprintf("list Close: %v", hresult_msg(hr)) }

	list := cast(^d3d12.ICommandList)g.list
	g.queue->ExecuteCommandLists(1, &list)
	// Flush here, not at wait time: the submit must reach the hardware
	// while the caller still has CPU work to overlap it with.
	return gpu_flush(g)
}

// dispatch_finish waits for a launched dispatch and reads the readback buffer
// into dst.
dispatch_finish :: proc(g: ^Gpu, readback: ^Buffer, dst: []u8) -> string {
	if err := gpu_join(g); err != "" { return err }
	return buffer_read(readback, dst)
}

// dispatch_and_read runs one compute dispatch with this package's root layout,
// then copies `out_bytes` from `dst` into `readback` and returns its bytes.
dispatch_and_read :: proc(
	g: ^Gpu,
	groups: [3]u32,
	constants: ^[ROOT_CONSTANT_COUNT]u32,
	srvs: [ROOT_SRV_COUNT]^Buffer,
	uav: ^Buffer,
	uav_before: d3d12.RESOURCE_STATES,
	dst: ^Buffer,
	readback: ^Buffer,
	out_bytes: int,
	dst_out: []u8,
) -> string {
	if err := dispatch_launch(g, groups, constants, srvs, uav, uav_before, dst, readback, out_bytes); err != "" {
		return err
	}
	return dispatch_finish(g, readback, dst_out)
}
