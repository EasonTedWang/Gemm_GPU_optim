# GEMM GPU 优化学习计划

本文档用于指导你从零开始学习 GEMM 的 GPU 优化，目标是从最基础的矩阵乘法实现，逐步过渡到接近 CUTLASS 风格的高性能 kernel 设计思路。

## 1. 学习目标

你最终希望达到的能力是：

- 理解 GEMM 的计算特点和性能瓶颈
- 了解 CUDA 线程、block、warp、SM、shared memory、register 的关系
- 从 naive kernel 开始，逐步实现 tiling、shared memory、warp-level 优化
- 学会用 Tensor Core 进行矩阵乘法加速
- 能用 Nsight Compute 等工具分析性能热点并持续优化
- 形成自己的一套“从 baseline 到高性能 kernel”的工程化方法

## 2. 推荐学习顺序

### 阶段 0：准备环境与基础认知（第 1-2 周）

目标：建立开发环境，掌握基本 CUDA 观念。

你需要完成：

- 安装 CUDA Toolkit、nvcc、Nsight Systems / Nsight Compute
- 了解 GPU 结构：SM、warp、shared memory、L2 cache、global memory
- 了解 GEMM 的数学形式：
  - $C = A \times B + C$
  - 计算量与访存量的关系
- 了解线程层次：
  - thread
  - block
  - grid
  - warp

建议产出：

- 一个能编译运行的最小 CUDA 程序
- 一个能打印 GPU 信息的测试程序

### 阶段 1：实现 CPU 参考版本与 Naive CUDA 版本（第 3 周）

目标：先建立“正确且可对比”的 baseline。

你要写的内容：

- CPU 版本 GEMM：用最直观的三重循环
- CUDA 版本 naive kernel：每个线程计算一个输出元素
- 对比 CPU 与 GPU 的结果正确性
- 记录不同矩阵规模下的性能

核心重点：

- 理解为什么 naive GPU 版本往往性能很差
- 观察访存不连续、global memory 带宽利用不高的问题

建议产出：

- 代码目录：src/01_naive
- 一个简单性能测试脚本

### 阶段 2：引入 tiling 与访存优化（第 4-5 周）

目标：提升 GPU 的计算与访存效率。

你要写的内容：

- 将矩阵分块（tile）
- 每个 block 负责一个输出 tile
- 使用 shared memory 缓存 A/B 的 tile
- 让线程在 block 内协作计算一个小块输出

核心重点：

- 线程块如何映射到输出 tile
- shared memory 如何减少 global memory 访问
- 如何让访存更连续、更高效

建议产出：

- 共享内存版本 GEMM
- 对比 naive 与 tiled 的性能提升

### 阶段 3：寄存器优化与 warp 级并行（第 6-7 周）

目标：进一步压榨每个 SM 的性能。

你要写的内容：

- 在每个线程中使用寄存器缓存中间结果
- 设计更合理的线程分配和 tiling 配形
- 尝试将每个 warp 处理更小的子块
- 研究共享内存的 bank conflict 以及如何规避

核心重点：

- occupancy 与 instruction-level parallelism
- register pressure 与 shared memory pressure 的权衡
- warp 内部的协作方式

建议产出：

- 一个更接近“手写优化 kernel”的版本
- 使用 Nsight Compute 观察 occupancy、SM throughput、memory throughput

### 阶段 4：Tensor Core 加速（第 8-10 周）

目标：真正接近现代高性能 GEMM 的实现思路。

你要写的内容：

- 使用 Tensor Core 的 mma 指令
- 设计合适的 warp tile 形状
- 处理 A/B/C 的布局与转置
- 了解 WMMA / CUDA 9+ 的 Tensor Core 编程模型

核心重点：

- Tensor Core 的工作方式
- 为什么它比普通 CUDA core 更适合 GEMM
- 为什么 tile shape、warp shape、instruction schedule 对性能影响巨大

建议产出：

- 一个 Tensor Core 版本 GEMM
- 对比普通 CUDA kernel 与 Tensor Core kernel 的性能

### 阶段 5：接近 CUTLASS 风格的工程化实现（第 11-12 周）

目标：不要求你一口气写出完整 CUTLASS，但要理解它的核心设计思想。

你要掌握：

- tile-based computation
- pipeline / double buffering
- warp-level tiling
- epilogue 处理
- layout 与 operand 配置
- kernel specialization 与模板化设计

你可以把你的实现逐步抽象为：

- tile shape 配置
- warp shape 配置
- instruction shape 配置
- epilogue 配置

建议产出：

- 一个结构化的 GEMM 内核框架，支持不同 tile 配置
- 形成你自己的“轻量版 CUTLASS 风格”实现

## 3. 推荐代码写作路径

建议按下面顺序落地代码：

### 第一步：建立项目骨架

建议创建以下目录：

- src/common/：公共工具与测试代码
- src/01_naive/：CPU 与 naive CUDA 版本
- src/02_tiled/：共享内存版
- src/03_warp/：warp 级优化版本
- src/04_tensor_core/：Tensor Core 版本
- scripts/：性能测试与绘图脚本

建议写几个基础文件：

- common/utils.h
- common/matrix_utils.cu
- common/benchmark.cu
- Makefile 或 CMakeLists.txt

### 第二步：每个阶段只做一个核心优化点

每次优化都要遵循同样的流程：

1. 先写出正确版本
2. 进行基准测试
3. 观察热点
4. 只改一个优化点
5. 再测一次性能
6. 记录结果与结论

不要一上来同时改很多东西，否则很难判断哪一步真的有效。

### 第三步：建立统一的测试与评估方式

每个版本都应至少包含：

- 数值正确性检查
- 不同形状下的性能测试
- 运行时间记录
- 吞吐量统计

建议使用以下矩阵规模做实验：

- 128x128
- 512x512
- 1024x1024
- 2048x2048

## 4. 推荐的学习与实践节奏

### 每周建议安排

- 3 天：阅读理论和代码
- 2 天：动手实现和调试
- 1 天：性能分析与总结
- 1 天：复盘与整理笔记

### 每个阶段建议输出

- 一份简短的笔记
- 一份可运行的代码
- 一段性能对比结果
- 一条你发现的优化结论

## 5. 建议的里程碑

- 里程碑 1：能正确运行 naive CUDA GEMM
- 里程碑 2：共享内存版本性能明显优于 naive
- 里程碑 3：能解释 occupancy、shared memory、bank conflict 等问题
- 里程碑 4：Tensor Core 版本正确运行并有明显加速
- 里程碑 5：形成自己的一套“高性能 GEMM 框架”思路

## 6. 你可以这样开始

建议第一步就从这三个任务开始：

1. 先写一个 CPU 版本 GEMM
2. 再写一个 naive CUDA 版本 GEMM
3. 再把它改成共享内存版并测性能

只要你把这三步真正跑通，你就已经具备了后续优化的基础。

## 7. 后续建议

如果你愿意，我下一步可以直接帮你做下面其中之一：

- 生成一个适合这个仓库的项目目录结构
- 先从 CPU 版本和 naive CUDA 版本开始写代码骨架
- 继续把后续的 tiled / shared memory / Tensor Core 版本逐步补齐
- 直接把一套可运行的 GEMM 优化实验框架写出来
