// Compute-only peak probe. Separate root signature and PSO from the GEMM, so
// the two can coexist on one device/queue.

package phonon_d3d12

import "core:fmt"
import "vendor:directx/d3d12"

FLOPS_DXIL: []u8 = #load("flops.dxil")

PEAK_THREADS :: 256

Peak :: struct {
	g:     ^Gpu,
	pso:   ^d3d12.IPipelineState,
	uav:   ^Buffer,
	state: d3d12.RESOURCE_STATES,
}

peak_create :: proc(g: ^Gpu) -> (p: ^Peak, err: string) {
	p = new(Peak)
	p.g = g

	// Reuse the GEMM root signature: its b0 block is 8 uints and u0 is bound at
	// ROOT_OUT_INDEX. The connection is only valid because both shaders use the
	// same b0/u0 registers, which peak_run documents.
	pso_desc := d3d12.COMPUTE_PIPELINE_STATE_DESC {
		pRootSignature = g.root_sig,
		CS             = d3d12.SHADER_BYTECODE{pShaderBytecode = raw_data(FLOPS_DXIL), BytecodeLength = d3d12.SIZE_T(len(FLOPS_DXIL))},
		NodeMask       = 0,
		CachedPSO      = {},
		Flags          = {},
	}
	pso_raw: rawptr
	hr := g.device->CreateComputePipelineState(&pso_desc, d3d12.IPipelineState_UUID, &pso_raw)
	if !hr_ok(hr) { peak_destroy(p); return nil, fmt.tprintf("peak CreateComputePipelineState: %v", hresult_msg(hr)) }
	p.pso = cast(^d3d12.IPipelineState)pso_raw

	if p.uav, err = buffer_create(g, u64(1 << 22), .DEFAULT, STATE_COMMON, {.ALLOW_UNORDERED_ACCESS}); err != "" {
		peak_destroy(p); return nil, err
	}
	p.state = STATE_COMMON
	return p, ""
}

// peak_run dispatches `groups` groups of PEAK_THREADS, each running `iters`
// iterations of 8 dot4 ops, and waits for the GPU.
peak_run :: proc(p: ^Peak, iters, groups: u32) -> string {
	g := p.g
	if hr := g.alloc->Reset(); !hr_ok(hr) { return fmt.tprintf("peak alloc Reset: %v", hresult_msg(hr)) }
	if hr := g.list->Reset(g.alloc, p.pso); !hr_ok(hr) { return fmt.tprintf("peak list Reset: %v", hresult_msg(hr)) }

	// A transition whose before/after states are equal is invalid, and after the
	// first run this buffer is already an unordered-access resource.
	if p.state != STATE_UNORDERED_ACCESS {
		transition(g.list, p.uav.res, p.state, STATE_UNORDERED_ACCESS)
		p.state = STATE_UNORDERED_ACCESS
	}

	constants := [ROOT_CONSTANT_COUNT]u32{iters, 0, 0, 0, 0, 0, 0, 0}
	g.list->SetComputeRootSignature(g.root_sig)
	g.list->SetPipelineState(p.pso)
	g.list->SetComputeRoot32BitConstants(ROOT_CONSTANTS_INDEX, ROOT_CONSTANT_COUNT, &constants, 0)
	g.list->SetComputeRootUnorderedAccessView(ROOT_OUT_INDEX, p.uav.res->GetGPUVirtualAddress())
	g.list->Dispatch(groups, 1, 1)

	if hr := g.list->Close(); !hr_ok(hr) { return fmt.tprintf("peak list Close: %v", hresult_msg(hr)) }
	list := cast(^d3d12.ICommandList)g.list
	g.queue->ExecuteCommandLists(1, &list)
	return gpu_wait(g)
}

peak_destroy :: proc(p: ^Peak) {
	if p == nil { return }
	buffer_destroy(p.uav)
	if p.pso != nil { p.pso->Release() }
	free(p)
}
