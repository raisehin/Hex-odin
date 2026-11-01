#ifndef HEX_TRANSCRIBE_SHIM_H
#define HEX_TRANSCRIBE_SHIM_H

// C ABI for native/transcribe_shim.cpp. Odin loads this as a DLL at runtime
// (core:dynlib), so hex.exe stays runnable before the C++ side is built.
#if defined(_WIN32)
#  define HEX_API __declspec(dllexport)
#else
#  define HEX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Returns a static version string, e.g. "transcribe.cpp 0.4.1 / ggml".
HEX_API const char *hex_shim_version(void);

// Loads a GGUF model once per process. 0 on success, non-zero on failure.
// On failure, hex_last_error() describes it.
HEX_API int hex_model_init(const char *model_path);

// Transcribes mono f32 samples at 16 kHz. Returns a malloc'd UTF-8 string
// (caller passes it back to hex_free) or NULL on failure.
HEX_API const char *hex_transcribe(const float *samples, int count, const char *language);

// Newline-separated compute backends (cpu, vulkan0, ...), malloc'd.
HEX_API const char *hex_list_devices(void);

// Human-readable error for the last failed call.
HEX_API const char *hex_last_error(void);

HEX_API void hex_free(const char *p);

// Frees the loaded ASR session; the next hex_model_init reloads it.
HEX_API void hex_model_close(void);

HEX_API int  hex_vad_init(const char *model_path);
HEX_API int  hex_vad_has_speech(const float *samples, int count);
HEX_API void hex_vad_free(void);

// CPU worker cap for the ASR session; 0 uses every usable core.
HEX_API void hex_set_threads(int n);

#ifdef __cplusplus
}
#endif

#endif
