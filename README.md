# Gemm_GPU_optim
利用 CUDA/AVX 等硬件从零开始优化 GEMM，借由 CODEX 来辅助学习

## 学习计划

本仓库的学习路径已经整理为文档，见 [docs/gemm_learning_plan.md](docs/gemm_learning_plan.md)。

建议按以下顺序推进：

1. 先理解 GEMM 的数学定义与 GPU 计算模型
2. 实现 CPU 参考版本和 naive CUDA 版本
3. 逐步引入 tiling、shared memory 与 warp 级优化
4. 最后进入 Tensor Core 和接近 CUTLASS 风格的结构化实现

## 项目结构

- `CMakeLists.txt` 根级 CMake 工程
- `code/` 源码目录
  - `code/origin_gemm.c` CPU GEMM 基线示例
  - `code/common/` 公共工具
  - `code/01_naive/` Naive CUDA GEMM 示例
  - `code/02_tiled/` Tiled GEMM 骨架
  - `code/03_warp/` Warp 级优化骨架
  - `code/04_tensor_core/` Tensor Core 骨架

## 如何运行

1. 创建构建目录：

```bash
mkdir -p build
cd build
```

2. 生成构建文件：

```bash
cmake .. -DENABLE_CUDA=ON
```

如果你没有安装 CUDA Toolkit，使用：

```bash
cmake .. -DENABLE_CUDA=OFF
```

3. 编译项目：

```bash
cmake --build . --target all_examples
```

或直接使用仓库根目录的一键脚本：

```bash
./build_all.sh
```

这样会生成以下所有可用示例，并把可执行文件输出到 `out/` 目录：
- `origin_gemm`
- `gemm_naive_cuda`
- `gemm_tiled_cuda`
- `gemm_warp_cuda`
- `gemm_tensor_core` (如果 CUDA 可用)

4. 运行程序：

```bash
./out/origin_gemm 256 256 256
```

如果你要运行 CUDA 示例：

```bash
./out/gemm_naive_cuda 256 256 256
./out/gemm_tiled_cuda 256 256 256
./out/gemm_warp_cuda 256 256 256
./out/gemm_tensor_core 256 256 256
```

4. 运行程序：

```bash
./code/origin_gemm 256 256 256
```

如果你已经编译了 CUDA 示例：

```bash
./code/gemm_naive_cuda 256 256 256
```

## 说明

当前仓库已搭建好 CMake 管理的基础工程结构，并提供 CPU GEMM 示例与 CUDA 示例骨架。你可以先从 `origin_gemm` 开始验证编译和运行，再逐步在 `code/01_naive`、`code/02_tiled` 等目录中实现优化版本。