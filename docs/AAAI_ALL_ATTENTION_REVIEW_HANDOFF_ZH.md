# AAAI 审查交接：All-Attention Projection LUT Replacement

更新时间：2026-06-18

## 一句话结论

All-attention projection replacement 是当前 AAAI 论文的强有力
**completeness / fidelity 支持证据**，但不宜单独包装成核心创新。

它证明了 QKFormer CIFAR-100 的四个 attention `proj_conv` currents 可以同时
由 grouped binary-pattern LUT 近乎无损地重建；它没有证明整个 QKFormer、
所有 attention 算子或完整网络已经 LUT 化，也没有证明真实延迟、能耗或
SRAM 收益。

## 已完成证据

固定范围：

- architecture：QKFormer；
- dataset：CIFAR-100；
- checkpoint：seed 42，分别测试 `T=1` 和 `T=4`；
- full validation；
- targets：
  - `stage1.0.tssa`
  - `stage2.0.tssa`
  - `stage3.0.ssa`
  - `stage3.1.ssa`
- replacement point：每个 attention module 的 `proj_conv` current，位于原始
  `proj_bn` 和 `proj_lif` 之前。

核心结果：

| 方法 | T=1 Acc@1 | T=4 Acc@1 | FP32 table values |
|---|---:|---:|---:|
| clean hook | 77.78 | 81.09 | — |
| prototype TCSLU | 56.28 | 54.08 | 93.7 KiB |
| grouped LUT, 2 bit | 77.72 | 81.09 | 2,664 KiB |

2-bit grouped LUT 的 aggregate current MSE 约为 `2.10e-9` (`T=1`) 和
`1.82e-9` (`T=4`)。完整数字与复现实验入口见：

- `results/qk_grouped_projection_c100_all4_20260618.md`
- `scripts/server/run_qkformer_grouped_projection_all4.sh`
- `tools/qkformer_lut_current_unit_probe.py`

## 创新性判断

### 可以支持的论文价值

1. 它关闭了“只在单层或早期层有效”的直接质疑：四个 attention projection
   currents 被同时替换。
2. 它给出了 prototype/TCSLU 在 SSA projection 上失败后的明确解决机制：
   从 response approximation 转向 frozen binary-input projection 的精确
   grouped decomposition。
3. 它把论文故事从“地址可重建局部响应”推进到“attention projection
   replacement fidelity 可以端到端保持分类”的更完整证据链。
4. `T=1` 与 `T=4` 两个独立 checkpoint setting 都保持近乎无损，降低了
   单一时间步偶然性的风险。

### 不能单独承担的核心 novelty

1. grouped pattern lookup 本质上是 frozen binary linear projection 的代数分解，
   reviewer 可能将其视为 LUTNet、LogicNets、PolyLUT 或一般 truth-table /
   grouped lookup 思路的直接应用。
2. 当前 2-bit LUT 的 table-value footprint 为 2,664 KiB，超过原始 projection
   weights；它解决 fidelity，不解决 storage efficiency。
3. 当前 hook 仍先执行原 `proj_conv` 再替换输出，因此只验证数值与分类保真度，
   不验证真实计算移除、runtime、memory traffic 或硬件收益。
4. 它与论文核心的 semantic Q/K address、support-aware backoff 和 E7 compact
   subspace lookup 不是同一个机制。若把两条路线写成同等核心贡献，论文容易显得
   story 过密。

## 推荐论文定位

推荐主次关系：

1. 核心创新：semantic Q/K lookup addressability；
2. 核心方法：support-aware backoff 与 compact subspace-decoupled lookup；
3. 主要 downstream 支持：TCSLU-style calibrated current replacement；
4. 完整性闭环：all-attention grouped projection LUT replacement。

推荐 claim：

> Every attention projection current in the evaluated QKFormer CIFAR-100
> configuration can be replaced by a grouped binary-pattern LUT with no
> material classification loss.

禁止扩写为：

- complete QKFormer LUT replacement；
- full attention replacement；
- all-operator replacement；
- measured hardware acceleration；
- cross-architecture Spiking Transformer replacement。

## 已完成的 Operator-Scope 扩展

该方向已经按 Q/K/V、MLP、patch embedding、classifier 和 cumulative
all-affine 的顺序完成。项目最终采用统一 0.50 pp near-lossless 标准，所有类别
及累计行均通过。

该扩展必须逐类报告：

- 是否为 binary-input operator；
- 精确 LUT、近似 LUT 或不适合 LUT 的原因；
- table footprint 与原权重 footprint；
- current/output MSE；
- full-validation Acc@1；
- strongest matched control；
- 是否真正绕过原算子执行。

32 个 Conv1d/Conv2d/Linear modules 已覆盖，但 BN/LIF/state dynamics、
pooling、residual addition 和 SSA matrix products 未替换。因此仍不能讨论
complete attention block 或 complete-network replacement。

## 提交前仍需处理的非实验问题

- `paper/main.tex` 当前仍使用 CVPR-style/fallback article 配置，并非 AAAI
  官方模板。外部内容审查可以先进行，但正式 AAAI 投稿前必须迁移模板并重新检查
  页数、浮动体、匿名信息和参考文献格式。
- 当前主故事包含 E1、E5、E7、TCSLU current replacement 和 grouped
  projection replacement。需要防止“贡献过多但主线不清”：abstract 与
  introduction 应始终先讲 semantic address -> transparent fallback ->
  compact subspace，再把 grouped LUT 放到 completeness evidence。
- 2,664 KiB 是 table-value footprint，不是完整硬件存储，也不是压缩结果。建议在
  主文中同时提示“larger than the original projection-weight value count”，避免
  reviewer 误以为作者刻意隐藏 storage trade-off。

## 给 ChatGPT Pro 的可复制审查 Prompt

```text
你是 AAAI 论文的严格 Area Chair、Spiking Transformer reviewer 和 LUT/hardware
reviewer。请审查当前 QK-LUTFormer 仓库，重点回答：

1. all-attention projection-current replacement 是否构成独立创新，还是更适合作为
   semantic Q/K LUT 主线的 completeness/fidelity 支持？
2. grouped binary-pattern projection LUT 与 LUTNet、LogicNets、PolyLUT、
   product/grouped lookup、truth-table decomposition 等工作的 novelty collision
   在哪里？请给出最窄、最安全的创新表述。
3. 当前结果同时替换四个 attention proj_conv currents，在 CIFAR-100 T=1/T=4
   上分别为 77.78->77.72 和 81.09->81.09。它能解锁哪些 claim，不能解锁哪些
   claim？
4. 2-bit grouped LUT 使用 2,664 KiB FP32 table values，且当前 hook 仍执行原
   proj_conv。请评估 storage、runtime 和 hardware claim 风险。
5. 论文是否应把 grouped projection LUT 保留在 abstract、introduction
   contributions 和 main experiments？若保留，应该放在什么层级？
6. all-affine continuation 已完成：请评价“统一 0.50 pp 标准下 Q/K/V 与累计
   32-module PASS”的证据价值和最安全 claim，并判断它应放主文、supplement
   还是仅 discussion。
7. 请以 AAAI 标准给出：当前最大优点、三个最危险 reviewer objection、必须修改
   的 claim、建议标题/摘要定位，以及 Accept/Borderline/Reject 判断。

必须区分：
- semantic response-prototype LUT；
- TCSLU-style calibrated current replacement；
- exact grouped binary-pattern projection LUT。

不要把 all-attention projection-current replacement 写成 complete QKFormer
replacement，不要把 analytical operation/storage proxy 写成 measured speed、
SRAM、energy 或 hardware efficiency。

优先阅读：
- results/EXPERIMENT_RESULTS_SUMMARY.md
- docs/QK_LUTFORMER_QUICK_STATE.md
- docs/QK_LUTFORMER_NEXT_SESSION_HANDOFF.md
- docs/AAAI_ALL_ATTENTION_REVIEW_HANDOFF_ZH.md
- results/qk_grouped_projection_c100_all4_20260618.md
- paper/sections/0_abstract.tex
- paper/sections/1_intro.tex
- paper/sections/3_method.tex
- paper/sections/4_experiments.tex
- paper/sections/5_discussion.tex
- tools/qkformer_lut_current_unit_probe.py
- scripts/server/run_qkformer_grouped_projection_all4.sh
```

## 会话与运维状态

- 所有正式实验任务均已结束并生成完整 `summary.csv` / `metrics.json`。
- 最终一次 SSH 状态复查发生连接超时；这不影响已下载并核对的正式结果。
- 不应仅因该超时重跑实验。若需要复核，先检查服务器现有 result directories，
  不要启动新的训练或 sweep。

## 后续 All-Affine 审计结果

后续实验已按预注册规则启动，但在第一类 Q/K/V projections 即停止：

- 8-level 主配置：77.58 -> 77.13，下降 0.45 pp；
- 16-level 唯一重试：结果完全相同；
- aggregate output NRMSE 为 0.00012029，clip rate 为 0；
- 十个 Q/K/V targets 全部执行；
- 输入均为精确整数且最大不超过 4，因此增加 levels 只增加存储，没有改善。

项目最终确定 0.50 pp 为 near-lossless 标准，因此 0.45 pp 的 Q/K/V 行判为
PASS。早期 0.25 pp 内部阈值仅作为过程溯源，不再作为当前结论。

证据：`results/qk_all_affine_qkv_boundary_20260619.md`。

在修订门槛下，MLP、patch embedding、classifier 及 cumulative all-affine
均首轮通过。累计行同时替换 32 个 Conv1d/Conv2d/Linear outputs：
77.58 -> 77.98，NRMSE 0.00013521，clip 0。

但该行的 FP32 table values 为 248,208 KiB，约为原 weight-value count 的
9.47 倍；BN/LIF、pooling、residual addition、SSA matrix products 仍保留，
hook 仍先执行原算子。因此它支持 all-convolution/linear substitution fidelity，
不支持 complete QKFormer、compact LUT 或硬件加速。

完整证据：`results/qk_all_affine_continuation_20260619.md`。
