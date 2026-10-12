# CLAUDE.md

先读 AGENTS.md、README.md、DEVELOPING.md，遵守相同约定。

自有代码按 [coding style](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md) 保持简单，使用完整单位命名、模块级导入和具体注解，短注释和清晰分支；保留外部配置与 checkpoint 键，不格式化第三方及官方原件。

训练入口：`recipes/qwen35-08b/train.py`，模型配方与固定配置也在同一目录；启动脚本：`recipes/qwen35-08b/train_qwen35_dp2.sh`。模型组件：`src/arch/`；流式压缩语料与必要 tokenizer 辅助：`src/data/`；分块 CE 与训练配置：`src/train/`；固定 Megatron 参考账本：`src/utils/megatron_budget.py`，官方原件与许可证：`third_party/megatron/`。固定 tokenizer 资产在 `recipes/qwen35-08b/tokenizer-qwen35/` 并忽略 Git。

使用 conda 的 olmo 环境。共享机器先创建自己的 branch/worktree，并配置个人提交身份和独立 hook 路径；代码通过 PR 合并 main。Ruff/pre-commit 只检查自有代码，不改官方或 vendored 原件。正式训练从干净 commit 创建冻结的运行 worktree。每次运行新目录，完整状态恢复，保留累计 FLOPs 和失败证据；`$RUNTIME_DIR` 默认位于 `$EXP_DIR/pretrain_exp-runtime`。

共享磁盘容量有限，运行前检查剩余空间。开发记录只保留必要的配置、简洁日志、结果、成本和校验摘要；checkpoint 保留当前恢复所需的最新完整状态和仍需验证的最终或候选状态。归档所需权重后，负责人确认没有运行或恢复依赖，再清理已被替代的 checkpoint，保留失败原因和成本记录，不动他人的文件。

CPU检查：`python -m pytest tests -q`、`pre-commit run --all-files`、`git diff --check`、`python -m recipes.qwen35-08b.train --help`。首次提交前先暂存文件再检查，避免漏掉未跟踪文件。训练和正式A评估的具体命令见 README.md；不要自行扩展预算或复用既有结果作为新训练证据。
