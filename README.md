# CUDA Kernel Baseline

一个适合写和验证 CUDA kernel 的最小工程骨架。

## 目录结构

```text
.
├── CMakeLists.txt
├── CMakePresets.json
├── Dockerfile
├── include/
│   └── cuda_utils.cuh
└── src/
    └── vector_add.cu
```

## 本地编译

前提：

- 安装 NVIDIA Driver
- 安装 CUDA Toolkit（需要 `nvcc`）
- 安装 CMake 3.24+
- 安装 Ninja（可选，但 `CMakePresets.json` 默认使用 Ninja）

```bash
cmake --preset release
cmake --build --preset release
./build/release/vector_add
```

## 指定 GPU 架构编译

如果你要在别的 NVIDIA GPU 上测试，可以显式指定架构：

```bash
cmake -S . -B build/sm80 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build/sm80
./build/sm80/vector_add
```

常见示例：

- `75` -> T4 / RTX 20 系列
- `80` -> A100
- `86` -> RTX 30 系列
- `89` -> RTX 40 部分型号
- `90` -> H100

## Docker 方式

如果目标机器已经安装好 NVIDIA Container Toolkit，可以直接：

```bash
docker build -t cuda-baseline .
docker run --rm -it --gpus all -v "$(pwd)":/workspace cuda-baseline
cmake --preset release
cmake --build --preset release
./build/release/vector_add
```

## 后续扩展建议

- 在 `src/` 下继续增加新的 `.cu` 可执行样例
- 把公共 kernel helper 放到 `include/`
- 如果后面要做性能实验，可以继续补 `benchmark/` 目录和 `cudaEvent`/Nsight 测试脚本

