# AGENTS.md

先读 README.md 和 DEVELOPING.md。本仓库维护 Qwen3.5-0.8B-Base 文本预训练及其 Engram 实验配方，使用 vendored OLMo-core；先 `import third_party`，不安装另一套 OLMo 覆盖它。

- 使用已有 `conda activate olmo`。共享环境升级需由维护者协调。
- 使用自己的 branch 和 worktree，`git config --worktree` 配置身份。每个功能单独提交，通过 PR 合并 main；首次提交由维护者完成。
- pre-commit 与 Ruff 检查自有代码。每个 worktree 用 `core.hooksPath --worktree` 指向个人 hook 目录；不覆盖共享 Git common hooks。初始仓库先暂存文件再检查，不能把空索引检查当作全仓通过。
- 自有代码遵循 [coding style](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md)：优先简单实现与完整单位命名，模块级导入、具体类型注解、短注释和清晰分支；配置与 checkpoint 外部键保持兼容，不引入单次调用抽象或通用框架。Ruff 检查注解与现代类型语法；第三方及官方原件不适用本次重构。
- 只使用 8 个官方压缩训练分片，流式读取，不清洗、过滤、去重、解压或拟合 tokenizer。现成 tokenizer 固定版本；训练随机初始化。
- 单机 DP2：global batch128，length8192，默认 microbatch16，梯度累积4；seed20261011。模型/optimizer/梯度保留 FP32，计算 BF16，FlashAttention2+FLA，完整激活重计算与1024-token分块CE。
- Qwen3.5 配方集中在 `recipes/qwen35-08b/`：`train.py`、`train_qwen35_dp2.sh`、`model.py`、`config.json`。固定下载的 tokenizer 放在其 `tokenizer-qwen35/`，忽略 Git；流式读取所需辅助归 `src/data/`。
- 四档训练预算3e17、1e18、3e18、1e19，仅调用 `src/utils/megatron_budget.py` 的固定版 Megatron 参考函数；官方原件和许可证在 `third_party/megatron/`。整数完整步误差≤1%；optimizer、通信、激活重计算不计入参考训练账本。FLA chunk额外矩阵计算的正式认证仍待确认。不要格式化或修改 vendored OLMo、Megatron参考原件及官方评估原件。
- Engram 配方位于 `recipes/qwen35-engram/`，组件在 `src/arch/memory/`，固定官方参考在 `third_party/engram/`。只适配单残差流和 DP，不宣称多流 HC、RL、offload 或表分片已实现。memory-only vocabulary map 必须从固定本地 tokenizer 准备，核对 backend/hash/EOS/pad，不修改原始数据、标签或分母。论文 gate 与 demo signed-sqrt gate 明确区分；表 LR×5/WD0、初始化等选择写入配置与证据。
- 固定 Megatron 参考没有 Engram 项。Engram 运行必须标记 `engram_experimental`，主干 Megatron 账本保持原值；附加投影/卷积主要项由 `src/utils/engram_budget.py` 单列，披露遗漏项，认证前不准入正式候选。不因表容量增大而使用 6ND 代替实际被执行的计算量。
- 正式运行需要干净 commit；工程 smoke 最多8个累计步，独立目录、源码文件哈希和成本留证。运行中不改所用源码，不覆盖输出，不清理别人的 GPU 进程。
- 当前四档使用 `--no-checkpoints --eval-cache`：不保存权重或训练状态，训练结束直接验证内存中的最终模型，复用既有两个rank的进程组。正式运行必须有匹配的完整A缓存；无checkpoint模式不能恢复，中断后该档从随机初始化重跑，历史失败成本保留。固定 seed 不代表 bitwise 保证。
- 可选checkpoint模式的恢复须保持模型、并行度、microbatch、数据、tokenizer与调度身份，加载全部模型/optimizer/trainer/RNG/数据状态；累计 FLOPs 不重置。
- A仅验证/选型，B不进入反馈。完整A用原始 UTF-8 分母；EOS、padding和未预测首token不能进入分母。部分A不报告完整BPB，native耗时不是正式EvalTime。
- 共享磁盘空间有限，启动前检查剩余容量。当前无checkpoint四档只记录必要的配置、简洁日志、训练/评估结果、权重摘要和全部成本；完整A缓存复用，不重复复制数据或大文件。若另用checkpoint模式，先预留完整状态空间，仅保留仍需恢复或验证的状态，由运行负责人确认无依赖后清理已被替代的文件，不删除其他人的文件。
- 保留失败、取消、重试的原因和全部成本记录。运行证据统一放在 `$EXP_DIR/pretrain_exp-runtime`，通过 `$RUNTIME_DIR` 传入，不加入 Git。docs仅放可公开的本模型四档图表和 JSONL，不带机器路径、账号、密钥或部署元数据。
- tests/为唯一项目测试目录，`python -m pytest tests -q`，使用临时小数据，不下载真实语料。仓库外文件路径通过环境变量传入。
