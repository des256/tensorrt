# The Measurements

Here are all measurements of all experiments.

## Llama 3.2 Instruct 3b

```
            Desktop          Jetson
           ONNX     TRT     ONNX      TRT
        CPU   CUDA       CPU    CUDA

f16     449   29    10   1830   X     X
q8f16   140   155   7    493    507   44
q8i8    149*  58*   6    507*   573*  47
q4f16   391^  72^   4    2075^  688^  30
q4i8    202^  73^   4    577^   697^  30

* = output quality degrades into repetition
^ = output ok, but shorter
```

### Findings

- at q8i8, the ONNX models are effectively useless
- but at more extreme quantization, this doesn't seem to matter so much
- this doesn't appear to be a problem for TensorRT
- in CPU, there is no preference f16 or f32, so f32 is a waste then
- in CPU, despite being more cache-friendly, q8 is faster than q4
- f32 does not run on the Jetson due to memory limit
- on Jetson, the CUDA path reverts back to CPU, possibly because quantized operations on ONNX are not well supported
- TensorRT blows everything out of the water
- q4i8 is the best deal with TensorRT (4ms Desktop, 30ms Jetson)
- Desktop is only around 4x faster than Jetson

### Future

- check memory usage during more complicated tests
- unsure what quantization does with STT and TTS, might need q4f16 or just raw f16 for STT encoder alone (audio signal is 16-bits)

## Parakeet v3
