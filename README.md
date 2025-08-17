# TRTLite

**TRTLite** is a lightweight, PyTorch-style wrapper for TensorRT that enables rapid construction of high-performance neural networks. 
It also supports compiling and integrating custom C++ plugins, making it easy to extend TensorRT for specialized operations.


## Motivation

years ago, to improve the efficiency of building TensorRT graphs and developing custom Plugins, we initiated this project with an initial design and concept. The core goal was to further abstract the TensorRT API while maintaining compatibility with PyTorch interfaces. This approach allows existing PyTorch inference code to be efficiently converted into TensorRT graphs. Additionally, leveraging Python greatly accelerated Plugin testing and feature alignment.


## Features

- **PyTorch-like Interface**: Build TensorRT networks using a familiar PyTorch-style API.
- **High Performance**: Optimized for fast inference on NVIDIA GPUs using TensorRT and custom plugins.
- **Custom C++ Plugins**: Compile and integrate your own C++ plugins seamlessly.
- **Lightweight and Easy to Use**: Minimal boilerplate, fast setup, and clear interface.


## Installation

Install via Poetry:

```bash
poetry add trtlite 
# pip install trtlite

# Or clone the repository and install dependencies:
git clone https://github.com/rzbv/trtlite.git
cd trtlite
poetry install
```

## Usage and Example

- **Simple**: [`./samples/simple`](./samples/simple)  
  A simple example to show the usage.

- **ResNet**: [`./samples/resnet`](./samples/resnet)  
  High-performance ResNet inference example using TRTLite.

- **YOLOv11**: [`./samples/yolo11`](./samples/yolo11)  
  Real-time object detection example demonstrating TRTLite efficiency on YOLOv11.

- **VisionTransformer**: [`./samples/vision_transformer`](./samples/vision_transformer)  
  A high effcient vision seq2seq implementation with custom encoder and decorder plugins.

- **Mask RCNN**: [`./samples/maskrcnn`](./samples/maskrcnn)  
  Instance segmentation example leveraging custom plugins and TensorRT optimizations.


## Dependencies
- numpy 1.23.5
- absl-py 2.3.1
- pycuda 2024.1

## Contributing

Contributions are welcome! You can submit issues or pull requests on GitHub:

## License

TRTLite is licensed under the Apache-2.0 License.