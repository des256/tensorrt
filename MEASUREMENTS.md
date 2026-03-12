# The Measurements

Here are all measurements of all experiments.

## Llama 3.2 Instruct 3b

```
        Desktop          Jetson
        ONNX        TRT  ONNX         TRT
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

### Verdict

- use TensorRT q4i8

## Parakeet v3

```
       Desktop            Jetson
       ONNX         TRT   ONNX       TRT
       CPU    CUDA        CPU  CUDA

f16    415    446   246
q8f16  386    426   244?
q8i8   704*   309*  246^
q4f16  400^   283^  247
q4i8   357    282^  251^

* = output missed entirely
^ = output incomplete
? = minor error
```

### Findings

- CUDA doesn't seem to improve much; possibly because this is a very CPU-heavy model (copying memory back and forth)
- could be interesting to explore rebuilding the model entirely to reduce the CPU/GPU interaction

### Verdict

- actually... let's forget about Parakeet, and move to Moonshine entirely

## Moonshine

```
       Desktop         Jetson
       ONNX       TRT  ONNX       TRT
       CPU  CUDA       CPU  CUDA

f16    101  181^  70*  483  484   24
q8f16  43*  83*   65*  163  247   24?
q8i8   91*  51*   6    X    194   12*
q4f16  70^  18^   65*  X    94    24^
q4i8   52^  17^   6    X    82    13*

* = output is garbage
^ = output incomplete
```

### Findings

- Moonshine is way faster than Parakeet
- the usable optimum for this kind of work for Moonshine is TensorRT f16. Quantization degrades the audio processing too much.

### Verdict

- Use TensorRT f16 (but fix NaN problem on Desktop)
