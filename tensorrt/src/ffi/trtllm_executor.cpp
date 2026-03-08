// C++ implementation of the flat C API defined in trtllm_executor.h.
// Wraps tensorrt_llm::executor::Executor.

#include "trtllm_executor.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <chrono>
#include <cstring>
#include <exception>
#include <optional>
#include <string>

namespace tle = tensorrt_llm::executor;

// ---------- thread-local error string ----------

static thread_local std::string g_last_error;

static void set_error(const char* msg) {
    g_last_error = msg;
}

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

// ---------- opaque handle is just the Executor ----------

struct TrtLlmExecutor {
    tle::Executor executor;

    TrtLlmExecutor(std::filesystem::path const& engine_dir,
                   tle::ModelType model_type,
                   tle::ExecutorConfig const& config)
        : executor(engine_dir, model_type, config) {}
};

// ---------- C API ----------

extern "C" {

const char* trtllm_get_last_error(void) {
    return g_last_error.c_str();
}

TrtLlmStatus trtllm_executor_create(
    const char* engine_dir,
    TrtLlmModelType model_type,
    float kv_cache_fraction,
    int32_t max_beam_width,
    TrtLlmExecutor** out_handle)
{
    try {
        // Register TRT-LLM plugins (safe to call multiple times).
        initTrtLlmPlugins();

        tle::ModelType mt;
        switch (model_type) {
            case TRTLLM_DECODER_ONLY:   mt = tle::ModelType::kDECODER_ONLY; break;
            case TRTLLM_ENCODER_ONLY:   mt = tle::ModelType::kENCODER_ONLY; break;
            case TRTLLM_ENCODER_DECODER: mt = tle::ModelType::kENCODER_DECODER; break;
            default:
                set_error("Invalid model type");
                return TRTLLM_ERROR;
        }

        tle::KvCacheConfig kv_config;
        if (kv_cache_fraction > 0.0f) {
            kv_config.setFreeGpuMemoryFraction(kv_cache_fraction);
        }

        tle::ExecutorConfig exec_config(max_beam_width);
        exec_config.setKvCacheConfig(kv_config);

        *out_handle = new TrtLlmExecutor(engine_dir, mt, exec_config);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_create");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_enqueue(
    TrtLlmExecutor* handle,
    const int32_t* tokens,
    size_t n_tokens,
    int32_t max_new,
    float temperature,
    int32_t top_k,
    float top_p,
    float rep_penalty,
    int32_t end_id,
    int32_t pad_id,
    int streaming,
    uint64_t* out_req_id)
{
    try {
        tle::VecTokens input_tokens(tokens, tokens + n_tokens);

        tle::SamplingConfig sampling;
        if (temperature > 0.0f) {
            sampling.setTemperature(temperature);
        }
        if (top_k > 0) {
            sampling.setTopK(top_k);
        }
        if (top_p > 0.0f) {
            sampling.setTopP(top_p);
        }
        if (rep_penalty > 0.0f) {
            sampling.setRepetitionPenalty(rep_penalty);
        }

        tle::OutputConfig output_config;
        output_config.excludeInputFromOutput = true;

        std::optional<int32_t> opt_end_id;
        if (end_id >= 0) opt_end_id = end_id;

        std::optional<int32_t> opt_pad_id;
        if (pad_id >= 0) opt_pad_id = pad_id;

        tle::Request request(
            std::move(input_tokens),
            max_new,
            streaming != 0,
            sampling,
            output_config,
            opt_end_id,
            opt_pad_id);

        *out_req_id = handle->executor.enqueueRequest(request);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_enqueue");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_await(
    TrtLlmExecutor* handle,
    uint64_t req_id,
    int32_t* out_tokens,
    size_t capacity,
    size_t* out_n_tokens,
    int* out_is_final,
    int64_t timeout_ms)
{
    try {
        std::optional<std::chrono::milliseconds> timeout;
        if (timeout_ms > 0) {
            timeout = std::chrono::milliseconds(timeout_ms);
        }

        auto responses = handle->executor.awaitResponses(req_id, timeout);

        if (responses.empty()) {
            *out_n_tokens = 0;
            *out_is_final = 0;
            return TRTLLM_OK;
        }

        // Take the last response (most recent).
        auto& response = responses.back();

        if (response.hasError()) {
            set_error(response.getErrorMsg());
            return TRTLLM_ERROR;
        }

        auto result = response.getResult();
        *out_is_final = result.isFinal ? 1 : 0;

        // Copy beam-0 output tokens.
        const auto& beam0 = result.outputTokenIds.at(0);
        size_t n = beam0.size();
        if (n > capacity) n = capacity;

        std::memcpy(out_tokens, beam0.data(), n * sizeof(int32_t));
        *out_n_tokens = n;

        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_await");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_cancel(TrtLlmExecutor* handle, uint64_t req_id) {
    try {
        handle->executor.cancelRequest(req_id);
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_cancel");
        return TRTLLM_ERROR;
    }
}

TrtLlmStatus trtllm_executor_shutdown(TrtLlmExecutor* handle) {
    try {
        handle->executor.shutdown();
        return TRTLLM_OK;
    } catch (const std::exception& e) {
        set_error(e.what());
        return TRTLLM_ERROR;
    } catch (...) {
        set_error("Unknown C++ exception in trtllm_executor_shutdown");
        return TRTLLM_ERROR;
    }
}

void trtllm_executor_destroy(TrtLlmExecutor* handle) {
    delete handle;
}

} // extern "C"
