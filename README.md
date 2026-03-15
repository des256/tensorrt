# tensorrt repo

Comparing CUDA and TensorRT on Jetson for STT, LLM and TTS implementations.

# Verdict

Clearly TensorRT and TensorRT-LLM are the way forward for Jetson and Nvidia cards.

On desktop, Llama 3.2 Instruct 3b default fp16 goes from 449ms TTFT to 4ms TTFT with quantization and TensorRT-LLM, which is a speedup of 112x. On Jetson, it goes from 493ms to 30ms, a speedup of 16x.

For Moonshine STT, desktop goes from 101ms TTFT to 6ms TTFT, a speedup of 17x. On the Jetosn, it goes from 483ms TTFT to 24ms TTFT, a speedup of 20x. But, quantization destroys the accuracy of the model, which makes sense, because audio is 16-bit.

Parakeet STT turns out to be not very GPU-friendly.

Pocket TTS pending...
