# Gemm_GPU_optim
利用CUDA/AVX等硬件从零开始优化GEMM，借由CODEX来辅助学习

## 学习计划

本仓库的学习路径已经整理为文档，见 [docs/gemm_learning_plan.md](docs/gemm_learning_plan.md)。

建议按以下顺序推进：

1. 先理解 GEMM 的数学定义与 GPU 计算模型
2. 实现 CPU 参考版本和 naive CUDA 版本
3. 逐步引入 tiling、shared memory 与 warp 级优化
4. 最后进入 Tensor Core 和接近 CUTLASS 风格的结构化实现

你可以从该文档开始，按阶段逐步完成代码与实验。