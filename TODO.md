-> trtllm-build works on jetson
-> compiling the TensorRT-LLM wheel on the host didn't work yet

- optimize LLM with TensorRT-LLM
- fuse TTS pipeline

# conversion

- download HF checkpoint for specific model
- convert checkpoint to something trtllm-build understands
- transfer checkpoints to jetson
- trtllm-build on jetson
-
