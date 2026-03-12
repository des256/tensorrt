// TDT argmax kernel: computes predicted token and duration index on GPU.
// Avoids downloading 8198 floats to host per decoder step.

#include <cstdint>
#include <cuda_runtime.h>

__global__ void kernel_tdt_argmax(
    const float* __restrict__ logits,
    int32_t* __restrict__ result,
    int vocab_size,
    int num_durations)
{
    // Token argmax over [0 .. vocab_size)
    int best_token = 0;
    float best_val = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > best_val) {
            best_val = logits[i];
            best_token = i;
        }
    }

    // Duration argmax over [vocab_size .. vocab_size + num_durations)
    int best_dur = 0;
    best_val = logits[vocab_size];
    for (int i = 1; i < num_durations; i++) {
        if (logits[vocab_size + i] > best_val) {
            best_val = logits[vocab_size + i];
            best_dur = i;
        }
    }

    result[0] = best_token;
    result[1] = best_dur;
    result[2] = (best_token == vocab_size - 1) ? 1 : 0;  // blank = last token
}

extern "C" void tdt_argmax(
    const float* logits,
    int32_t* result,
    cudaStream_t stream,
    int vocab_size,
    int num_durations)
{
    kernel_tdt_argmax<<<1, 1, 0, stream>>>(
        logits, result, vocab_size, num_durations);
}
