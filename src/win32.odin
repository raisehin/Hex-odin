package main

import "core:sys/windows"

foreign import kernel32 "system:kernel32.lib"
@(default_calling_convention = "system")
foreign kernel32 {
	GetLocalTime           :: proc(st: ^windows.SYSTEMTIME) ---
	GetActiveProcessorCount :: proc(group: u32) -> u32 ---
	SetProcessInformation  :: proc(hProcess: windows.HANDLE, class: i32, info: rawptr, size: u32) -> windows.BOOL ---
}

// EcoQoS: Windows 11 puts the process on efficiency cores and caps clocks.
// On while the HUD is hidden (idle dictation host), off the moment a dictation
// starts so transcription keeps full speed.
ALL_PROCESSOR_GROUPS :: 0xFFFF

// Logical CPUs visible to the process. Used to pick a ggml thread count: see
// worker_threads.
logical_cpu_count :: proc() -> int {
	n := int(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS))
	return n > 0 ? n : 1
}

PROCESS_POWER_THROTTLING_STATE :: struct {
	Version:     u32,
	ControlMask: u32,
	StateMask:   u32,
}

ProcessPowerThrottling                  :: 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION :: 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED :: 0x1

power_eco :: proc(on: bool) {
	st := PROCESS_POWER_THROTTLING_STATE{
		Version     = PROCESS_POWER_THROTTLING_CURRENT_VERSION,
		ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
		StateMask   = on ? PROCESS_POWER_THROTTLING_EXECUTION_SPEED : 0,
	}
	SetProcessInformation(windows.GetCurrentProcess(), ProcessPowerThrottling, &st, size_of(st))
}
