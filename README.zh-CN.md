# RWKV-7 (Goose) × sglang

[English](README.md) · **简体中文**

**RWKV-7 在 [sglang](https://github.com/sgl-project/sglang) 上的生产级推理**:
输出与参考实现逐 token 一致、支持 int8/int4 量化、覆盖 11 种平台——10 种 CUDA GPU
(从 2018 年的 T4 到 B200、RTX 5090)加 Apple Silicon。
下文每一个数字的原始日志都在 [`bench/results/`](bench/results/) 里。

**➡ 完整基准文档:[docs/BENCHMARKS.zh-CN.md](docs/BENCHMARKS.zh-CN.md)**——所有测过的
指标轴,可读的表格形式(正确性验证、精度指标、逐卡速度、与 Albatross 的对比、量化
取舍、真实负载延迟),每张表链接到原始日志。

**同时支持 sglang `main` 和 v0.5.10**——同一份代码,版本差异在运行时自动识别。
模型支持的核心部分已提交上游:[sglang PR #30115](https://github.com/sgl-project/sglang/pull/30115)。

## 为什么用 RWKV-7 做推理服务

RWKV-7 是循环模型:每条序列的状态是**固定大小**的,和上下文多长无关——而 Transformer 的
KV 缓存随 token 数线性增长。实测效果:并发从 1 路加到 256 路、或上下文拉长 64 倍,
显存各只多用 **不到 0.2 GB**。高并发、长上下文正是这个架构的优势区。

## 目前能做什么(2026-07-05)

| | |
|---|---|
| **正确性** | 贪心输出与 numpy fp32 参考实现逐 token 一致——0.1B / 1.5B / 7.2B(CUDA)和 Apple Silicon(MLX)都是 24/24;动态批、分块预填充、CUDA graph、TP 2/4/8、PP 2/4/8 下同样精确 |
| **精度指标** | MATH500 贪心:新版 0.3940,旧版 0.3920——差异在统计波动内,精度没有变化。压缩率:fp16 0.6085,int8 0.6086(即无损) |
| **服务能力** | 动态批处理、分块预填充、循环状态前缀缓存(高复用负载下命中率约 98%) |
| **量化** | 两档 int8 + int4(GPTQ),全手写 CUDA。**w8g64**(仅权重):贪心无损(oracle 24/24 逐位)。**w8a8**(tensor-core):压缩率与 cutlass 一致(0.6161),MATH500 avg@64 实测 −2.3pt,换大批量/省显存。sm120/Blackwell 上——上游根本没有 int8 GEMM——我们自写 s8-wmma 核跑通 w8a8 且 GEMM 快过 fp16 cuBLAS(批 ≥512 时 1.03–1.55×)。**7.2B / 单张 32GB 5090 上 int8 并发 2.90×、峰值高 26.8%**(fp16 超 221 并发就 OOM)——详见 [BENCHMARKS §4](docs/BENCHMARKS.zh-CN.md) / [F0035](docs/findings/0035-7b-int8-concurrency-headroom.md) |
| **投机解码** | 第一阶段已跑通:小模型起草、大模型一次验证,拒绝的部分靠固定大小的状态快照回滚。10 组对照测试中 9 组与普通解码逐 token 相同,唯一差异查明是浮点运算顺序的极小抖动,不是算法错误——完整分析见 [F0031](docs/findings/0031-spec-decode-increment-i.md) |
| **Apple Silicon** | 原生 MLX 实现,自写 Metal 计算核,用同一个 numpy 参考做验证——见 [`mlx_port/`](mlx_port/) |
| **上游贡献** | 模型 PR [#30115](https://github.com/sgl-project/sglang/pull/30115)(在 RTX 3090 与 RTX 5090 上验证);另发现并修复了 sglang 流水线并行的一个静默数据损坏:issue [#30015](https://github.com/sgl-project/sglang/issues/30015) → 修复 PR [#30095](https://github.com/sgl-project/sglang/pull/30095) |

## 速度

1.5B 模型,单卡,sglang main。"单请求" = 一条流的持续解码速度;
"峰值" = 并发扫描中的最佳总吞吐(64 token 提示词、256 token 输出)。

| GPU(1.5B) | 单请求 | 峰值服务吞吐 |
|---|---|---|
| RTX 3090 | 230.7 tok/s | 7,205 tok/s(fp16)· **9,851 tok/s(int8)** |
| RTX 5090 | **409.8 tok/s**(fp16)· **548.8**(int4) | **22,175 tok/s** |

**7.2B,单张 RTX 5090(32GB):** 单请求 123.7 tok/s(fp16)。峰值服务:5,983 tok/s
(fp16,但只能到 221 并发,再高就 OOM)对 **7,587 tok/s(int8,640 并发)**。bsz1 时 fp16
更快;int8 的优势是显存余量让 7.2B 扩到 2.90× 并发——完整对比见
[BENCHMARKS §4](docs/BENCHMARKS.zh-CN.md)。

- 同一套代码在 T4、L4、A10G、A100(40/80GB)、L40S、H100、H200、B200 上全部正常运行——
  逐卡数据见 [`fleet_main_10cards.json`](bench/results/fleet_main_10cards.json)。
- 真实负载抽样(ShareGPT 对话数据,RTX 5090):峰值输出 9,845 tok/s;每秒 16 个请求到达时,
  首字延迟中位数 32 毫秒。
- **与 BlinkDL 的 Albatross 对比**(官方速度参照;注意它是一个纯测速程序,没有请求调度和
  服务接口):我们的单请求速度是它的 0.9004 倍(L4)到 0.5129 倍(B200)不等——GPU 显存带宽
  越高,它的整层融合设计越占优。在作者本人调参用的 RTX 5090 上,我们的 int4 达到它 fp16 速度的
  **0.9908 倍**。T4 这类老卡上 Albatross 无法编译(它用了 sm80 以上才有的指令),我们可以正常
  服务。逐卡数据:[`albatross_fleet_10cards.json`](bench/results/albatross_fleet_10cards.json)。

## 快速上手

**在 sglang main 上**(例如 `lmsysorg/sglang:dev-cu12` 容器内):

```bash
cd /sgl-workspace/sglang
git apply <本仓库>/sglang_main_port/upstream_edits.patch   # 7 处小的接线修改
# 然后复制 RWKV-7 文件(模型、后端、计算核、配置):
#   文件清单和目标路径见 sglang_main_port/README.md
python -m sglang.launch_server --model-path <rwkv7模型目录> --trust-remote-code \
    --attention-backend triton --dtype float16 --disable-radix-cache
```

**在 sglang v0.5.10 上**(pip 安装的环境):`BOX=<主机> SP=<site-packages路径> bash scripts/deploy.sh`
——rsync 覆盖层并应用两处单行补丁。

手写加速核通过环境变量按需开启,全部经过贪心精确验证;推荐的生产组合见
[`scripts/serve.sh`](scripts/serve.sh)。模型:任意 fla 格式的 RWKV-7 权重
(HF 上的 `fla-hub/rwkv7-*`),或我们发布在 ModelScope 的 int8/int4 量化权重(`Hakureirm/rwkv7-g1-*`)。

**在 Mac 上**:见 [`mlx_port/README.md`](mlx_port/README.md)。

## 目录结构

```
sglang_overlay/    实现本体:模型、状态后端、CUDA/Triton 计算核、投机解码 worker
sglang_main_port/  同一份代码在 sglang main 上的应用方式(补丁 + 文件清单)
mlx_port/          Apple Silicon 原生实现(MLX + Metal 核)
bench/             全部基准与正确性验证脚本;原始输出在 bench/results/
docs/              编号的测量报告(findings)与设计决策(ADR)——完整证据链
scripts/           deploy.sh(v0.5.10 部署)· serve.sh(推荐启动参数)
```

## 每个数字的出处

[`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) 把每个关键数字对应到它的原始日志。
[`docs/findings/`](docs/findings/) 是带日期的测量报告,方法学齐全,负面结果也如实记录。
如果你重跑 `bench/` 里的脚本得到不同的数字,欢迎提 issue。
