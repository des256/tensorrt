# The Measurements

Here are the measurements of all experiments.

## Llama 3.2 Instruct 3b

```
        Desktop          Jetson
        ONNX        TRT  ONNX         TRT
        CPU   CUDA       CPU    CUDA

f16     449   29    10   1830
q8f16   140   155   7    493
q8i8    149*  58*   6    507*
q4f16   391^  72^   4    2075^
q4i8    202^  73^   4    577^

* = output quality degrades into repetition
^ = output ok, but shorter
```

### Findings

- at q8i8, the ONNX models are effectively useless.
- but at more extreme quantization, this doesn't seem to matter so much
- this doesn't appear to be a problem for TensorRT-LLM
- in CPU, there is no preference f16 or f32
- in CPU, q8 is faster than q4, probably due to an extra decoding step
- f32 does not run on CUDA on the Jetson
- --> need to rebuild the engines on the Jetson, then run the tests
