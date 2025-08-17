
## Model Brief

A Very Small Vision Transformer

Model Archtecture:

- VIT: RESNET-34
- PROJECT: CONV+FC
- Transformer: 
  - 3 x Encoder Layer (768 hidden size, 8 header, 64 head size)
  - 3 x Decoder Layer (768 hidden size, 8 header , 64 head size)

Total Parameters: ~50 Millions.

average input tokens: 200, average output tokens: 100.

## Perfomance Test

Hardware:  NVIDIA GeForce RTX 3080
Software Stack: TensorRT 10.1 + CUDA 12.5


| Framework  | Speed (ms/seq) |
|------------|-------------------|
| PyTorch    | ~15 ms/seq        |
| TensorR+Plugins | ~ 1.2 ms/seq |
