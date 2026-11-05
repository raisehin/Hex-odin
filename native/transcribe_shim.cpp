// C ABI bridge between Odin and transcribe.cpp (ggml). Replaces the Rust
// `transcribe-cpp` crate glue: model/session lifetime and error text.
//
// Links the checkout at native/third_party/transcribe.cpp (pinned rev
// 7045d1c0fe7cee37747b2792977421d435e75fca); see native/CMakeLists.txt.

#include "transcribe_shim.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#ifdef HEX_HAVE_TRANSCRIBE
#include <transcribe.h>
#include <transcribe/vad.h>
#endif

namespace {

std::mutex g_mutex;
std::string g_error;
#ifdef HEX_HAVE_TRANSCRIBE
transcribe_session *g_session = nullptr;
transcribe_model   *g_vad_model = nullptr;
int g_threads = 0; // 0 = library default (all usable cores)
transcribe_vad_session *g_vad_session = nullptr;
#endif
std::string g_model_path;

const char *dup(const std::string &s) {
    char *p = static_cast<char *>(std::malloc(s.size() + 1));
    if (!p) return nullptr;
    std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

void set_error(const std::string &e) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_error = e;
}

} // namespace

extern "C" const char *hex_shim_version(void) {
#ifdef HEX_HAVE_TRANSCRIBE
    static std::string version;
    if (version.empty()) {
        const char *lib = transcribe_version ? transcribe_version() : "?";
        const char *commit = transcribe_version_commit ? transcribe_version_commit() : "?";
        version = std::string("transcribe.cpp ") + lib + " (" + commit + ")";
    }
    return version.c_str();
#else
    return "transcribe_shim 0.1.0 (stub: rebuild with transcribe.cpp)";
#endif
}

extern "C" int hex_model_init(const char *model_path) {
#ifdef HEX_HAVE_TRANSCRIBE
    if (!model_path || !*model_path) { set_error("empty model path"); return 1; }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_session && g_model_path == model_path) return 0;
    if (g_session) { transcribe_session_free(g_session); g_session = nullptr; }

    transcribe_session *session = nullptr;
    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    sp.n_threads = g_threads;
    transcribe_status status = transcribe_open(model_path, nullptr, &sp, &session);
    if (status != TRANSCRIBE_OK || !session) {
        g_error = transcribe_status_string(status);
        return 2;
    }
    g_session = session;
    g_model_path = model_path;
    g_error.clear();
    // transcribe.cpp logs by architecture ("parakeet:", "canary:"), which is
    // confusing when the file is Phonon-2.5 (a Parakeet-family checkpoint).
    std::fprintf(stderr, "[hex] model file: %s\n", model_path);
    return 0;
#else
    (void)model_path;
    set_error("built without transcribe.cpp: run scripts/build-native.ps1");
    return 3;
#endif
}

// Caps ggml's CPU workers. A change drops the cached session so the next
// transcription reopens with the new count; 0 restores the library default.
extern "C" void hex_set_threads(int n) {
#ifdef HEX_HAVE_TRANSCRIBE
    std::lock_guard<std::mutex> lock(g_mutex);
    if (n < 0) n = 0;
    if (n == g_threads) return;
    g_threads = n;
    if (g_session) {
        transcribe_session_free(g_session);
        g_session = nullptr;
        g_model_path.clear();
    }
#else
    (void)n;
#endif
}

extern "C" const char *hex_transcribe(const float *samples, int count, const char *language) {
#ifdef HEX_HAVE_TRANSCRIBE
    if (!samples || count <= 0) { set_error("no samples"); return nullptr; }

    // A language hint is what makes multilingual checkpoints (canary, whisper,
    // qwen3-asr) resolve the right tokenizer state; NULL means autodetect.
    transcribe_run_params params;
    transcribe_run_params_init(&params);
    if (language && language[0]) params.language = language;

    // Checked under the lock so an idle release cannot free the session
    // between the check and the run.
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_session) { g_error = "model not loaded"; return nullptr; }
    transcribe_status status = transcribe_run(g_session, samples, count, &params);
    if (status != TRANSCRIBE_OK) {
        g_error = transcribe_status_string(status);
        return nullptr;
    }
    const char *text = transcribe_full_text(g_session);
    return dup(text ? text : "");
#else
    (void)samples; (void)count; (void)language;
    set_error("built without transcribe.cpp");
    return nullptr;
#endif
}

extern "C" const char *hex_list_devices(void) {
#ifdef HEX_HAVE_TRANSCRIBE
    std::string out = "transcribe.cpp ";
    out += transcribe_version ? transcribe_version() : "?";
    out += "\n";
    const char *commit = transcribe_version_commit ? transcribe_version_commit() : "?";
    out += "commit ";
    out += commit;
    return dup(out);
#else
    return dup("stub: transcribe.cpp not linked");
#endif
}

extern "C" const char *hex_last_error(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_error.c_str();
}

// Releases the ASR session (and its ~700 MB of weights) without touching the
// VAD model. The next hex_model_init reopens it.
extern "C" void hex_model_close(void) {
#ifdef HEX_HAVE_TRANSCRIBE
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_session) {
        transcribe_session_free(g_session);
        g_session = nullptr;
    }
    g_model_path.clear();
#endif
}

extern "C" void hex_free(const char *p) {
    std::free(const_cast<char *>(p));
}

// ---------------------------------------------------------------------------
// VAD (Silero): gate transcription on actual speech so silence never reaches
// the ASR model, which otherwise invents closing phrases.
// ---------------------------------------------------------------------------

extern "C" int hex_vad_init(const char *model_path) {
#ifdef HEX_HAVE_TRANSCRIBE
    if (!model_path) return 1;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_vad_session) return 0;
    transcribe_model *model = nullptr;
    if (transcribe_model_load_file(model_path, nullptr, &model) != TRANSCRIBE_OK) {
        // Already holding g_mutex here; set_error() would re-lock the same
        // non-recursive mutex and abort the process.
        g_error = "vad model load failed";
        return 1;
    }
    transcribe_vad_session_params sp;
    transcribe_vad_session_params_init(&sp);
    transcribe_vad_session *session = nullptr;
    if (transcribe_vad_session_init(model, &sp, &session) != TRANSCRIBE_OK) {
        transcribe_model_free(model);
        g_error = "vad session init failed";
        return 2;
    }
    g_vad_model = model;
    g_vad_session = session;
    g_error.clear();
    return 0;
#else
    (void)model_path;
    return 3;
#endif
}

// 1 = speech found, 0 = silence, negative = VAD unavailable/failed.
extern "C" int hex_vad_has_speech(const float *samples, int count) {
#ifdef HEX_HAVE_TRANSCRIBE
    if (!g_vad_session) return -1;
    if (!samples || count <= 0) return 0;
    std::lock_guard<std::mutex> lock(g_mutex);
    transcribe_vad_params params;
    transcribe_vad_params_init(&params);
    if (transcribe_vad_run(g_vad_session, samples, count, &params) != TRANSCRIBE_OK) return -1;
    transcribe_vad_result result;
    transcribe_vad_result_init(&result);
    if (transcribe_vad_get_result(g_vad_session, &result) != TRANSCRIBE_OK) return -1;
    return result.n_segments > 0 ? 1 : 0;
#else
    (void)samples; (void)count;
    return -1;
#endif
}

// End sample of the last speech segment that is already CLOSED in this audio,
// i.e. it is followed by at least min_silence_ms of audio, so splicing there
// cannot cut a word in half. Returns -1 when nothing is closed yet or the VAD
// is unavailable, which tells the caller to keep accumulating.
//
// The VAD already computes these boundaries for hex_vad_has_speech and threw
// them away; this exposes them so the caller can transcribe a finished segment
// while the user is still talking, instead of encoding the whole dictation
// after key-up.
extern "C" long long hex_vad_speech_end(const float *samples, int count, int min_silence_ms) {
#ifdef HEX_HAVE_TRANSCRIBE
    if (!g_vad_session || !samples || count <= 0) return -1;
    std::lock_guard<std::mutex> lock(g_mutex);
    transcribe_vad_params params;
    transcribe_vad_params_init(&params);
    if (min_silence_ms <= 0) min_silence_ms = params.min_silence_ms;
    if (transcribe_vad_run(g_vad_session, samples, count, &params) != TRANSCRIBE_OK) return -1;
    transcribe_vad_result result;
    transcribe_vad_result_init(&result);
    if (transcribe_vad_get_result(g_vad_session, &result) != TRANSCRIBE_OK) return -1;
    // 16 kHz: the VAD session runs at the model rate.
    const long long closed_before = (long long)count - (long long)min_silence_ms * 16LL;
    long long best = -1;
    for (int i = 0; i < result.n_segments; ++i) {
        transcribe_vad_segment seg;
        transcribe_vad_segment_init(&seg);
        if (transcribe_vad_get_segment(g_vad_session, i, &seg) != TRANSCRIBE_OK) continue;
        if (seg.end_sample > closed_before) continue; // still open, speech may resume
        if (seg.end_sample > best) best = seg.end_sample;
    }
    return best;
#else
    (void)samples; (void)count; (void)min_silence_ms;
    return -1;
#endif
}

extern "C" void hex_vad_free(void) {
#ifdef HEX_HAVE_TRANSCRIBE
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_vad_session) { transcribe_vad_session_free(g_vad_session); g_vad_session = nullptr; }
    if (g_vad_model) { transcribe_model_free(g_vad_model); g_vad_model = nullptr; }
#endif
}
