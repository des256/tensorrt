6# MODEL DISCOVERY

## TESTS

To test a model and find the ultimate version for the project:

on desktop, from model source, export the following ONNX models:

- full f32 weights and data path to `onnx/f32`
- full fp16 weights and data path to `onnx/f16`
- quantized int8 weights and int8 data path to `onnx/q8i8`
- quantized int8 weights and fp16 data path to `onnx/q8f16`
- quantized 4-bit weights and int8 data path to `onnx/q4i8`
- quantized 4-bit weights and fp16 data path to `onnx/q4f16`

inside docker container, from model source, export the following TensorRT(-LLM) checkpoints:

- full fp16 weights and data path to `ckpt/f16`
- quantized int8 weights and int8 data path to `ckpt/q8i8`
- quantized int8 weights and fp16 data path to `ckpt/q8f16`
- quantized 4-bit weights and int8 data path to `ckpt/q4i8`
- quantized 4-bit weights and fp16 data path to `ckpt/q4f16`

inside docker container, from the TensorRT(-LLM) checkpoints, build the following desktop engines:

- full fp16 weights and data path to `engine/desktop/f16`
- quantized int8 weights and int8 data path to `engine/desktop/q8i8`
- quantized int8 weights and fp16 data path to `engine/desktop/q8f16`
- quantized 4-bit weights and int8 data path to `engine/desktop/q4i8`
- quantized 4-bit weights and fp16 data path to `engine/desktop/q4f16`

outside docker container, run the following tests:

- run all ONNX tests with Cpu provider on Desktop
- run all ONNX tests with Cuda provider on Desktop
- run all TensorRT(-LLM) tests on Desktop

on jetson, from the TensorRT(-LLM) checkpoints, build the following jetson engines:

- full fp16 weights and data path to `engine/jetson/f16`
- quantized int8 weights and int8 data path to `engine/jetson/q8i8`
- quantized int8 weights and fp16 data path to `engine/jetson/q8f16`
- quantized 4-bit weights and int8 data path to `engine/jetson/q4i8`
- quantized 4-bit weights and fp16 data path to `engine/jetson/q4f16`

then run the following tests:

- run all ONNX tests with Cpu provider on Jetson
- run all ONNX tests with Cuda provider on Jetson
- run all TensorRT(-LLM) tests on Jetson

## PRODUCTION

Later, once we know what the best options are for a model:

### ONNX

on desktop, from model source, export the ONNX models in desired format and save it in `data/{}`

### TENSORRT(-LLM)

on desktop, in docker container, from model source, export the TensorRT(-LLM) checkpoint

on desktop, in docker container, from the checkpoint, build the corresponding engine and save it in `data/{}/89`

on jetson, from the checkpoint, build the corresponding engine and save it in `data/{}/87`
