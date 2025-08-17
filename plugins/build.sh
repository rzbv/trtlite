#!/bin/bash

function build() {
      current_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
      cd $current_dir

      TRT_LIB=/usr/lib/x86_64-linux-gnu
      TRT_INC=/usr/include/x86_64-linux-gnu
      CUDNN_INC="/usr/include/x86_64-linux-gnu"
      CUDNN_LIB="/usr/lib/x86_64-linux-gnu"

      if [[ ! -d build ]]; then
            mkdir -p build
      fi
      cd build
      cmake ../src \
            -DTRT_OUT_DIR="${current_dir}/output" \
            -DCUDA_VERSION=12.5 -DCUDNN_VERSION=9.1 \
            -DBUILD_PLUGINS=ON -DBUILD_SAMPLES=ON \
            -DTRT_LIB_DIR=$TRT_LIB -DTRT_INC_DIR=$TRT_INC \
            -DCUDNN_INC_DIR=$CUDNN_INC -DCUDNN_LIB_DIR=$CUDNN_LIB

      make -j$(nproc)
      
      # release the library
      mkdir -p /usr/local/trtlite/lib/ || echo "escape"
      cd $current_dir 
      cp -fL ./output/libnvinfer_plugin.so /usr/local/trtlite/lib/libtrtlite_plugin.so
}

build
