# 共享机器开发流程

维护 `main` 的人先完成首次提交并推送。首次提交以前不能创建分支 worktree，也不能启动正式训练。首次提交包含当前文件即可，不导入其他仓库的 Git 历史。维护者可将 `main` 设置为只允许通过 PR 合并，并要求检查通过和至少一名 reviewer 批准。

维护者先按下文配置自己的身份与个人 hook，再在已有仓库中执行：

```bash
conda activate olmo
git add .
pre-commit run --all-files
python -m pytest tests -q
git diff --cached
# 格式工具若有改动，核对后再次暂存、检查。
git add .
pre-commit run --all-files
git commit -m "Initialize Qwen3.5 pretraining"
git push git@github.com:kkyyxhll/pretrain_exp.git main
```

`--all-files` 检查 Git 跟踪的文件；初始空索引必须先暂存，不能将未检查新文件的结果当作通过。忽略的 tokenizer 下载资产、数据、runtime 和缓存不应加入提交。

## 创建自己的分支和 worktree

在主仓库中执行；替换名字和功能名，每个功能使用一个分支：

```bash
cd "$EXP_DIR/pretrain_exp"
git config extensions.worktreeConfig true
export GIT_SSH_COMMAND="ssh -i /private/path/to/your/key -o IdentitiesOnly=yes"
git fetch git@github.com:kkyyxhll/pretrain_exp.git main:refs/remotes/origin/main
mkdir -p "$EXP_DIR/worktrees"
git worktree add -b alice/qwen35-loader "$EXP_DIR/worktrees/alice-qwen35-loader" origin/main
cd "$EXP_DIR/worktrees/alice-qwen35-loader"
```

`git worktree list` 查看已占用的路径和分支。每个 worktree 有自己的工作目录、索引和 HEAD；对象库、remote、refs 默认共享。不要多人直接编辑主仓库或同一 worktree，不要在别人的 worktree 执行 checkout、reset、clean、stash 或 rebase。

## 配置自己的提交身份和登录

先设置自己的状态目录，存放 GitHub CLI、pre-commit 缓存和 hook；该目录位于仓库外，名称由本人选择：

```bash
export PERSONAL_STATE_DIR="$EXP_DIR/personal-state/alice"
mkdir -p "$PERSONAL_STATE_DIR"
chmod 700 "$PERSONAL_STATE_DIR"
```

只设置当前 worktree；不要修改共享的 global/local 身份配置：

```bash
git config --worktree user.name "Your Name"
git config --worktree user.email "YOUR_VERIFIED_EMAIL_OR_GITHUB_NOREPLY"
git config --worktree user.useConfigOnly true
git var GIT_AUTHOR_IDENT
git config --show-origin --get user.email
```

提交身份不等于 GitHub 登录身份。优先使用各自的 Linux 用户和个人 SSH key；同一 Linux 账号时，每个人的 SSH key、SSH agent、GitHub CLI 配置与凭证缓存应使用独立路径。不要共用 root 的默认凭证，也不要把 token 写入 Git URL、仓库文件或命令参数。

个人 SSH key 位于仓库外，在自己的 worktree 绑定它（替换绝对路径）：

```bash
git config --worktree core.sshCommand "ssh -i /private/path/to/your/key -o IdentitiesOnly=yes"
unset GIT_SSH_COMMAND
# 只对本次 push 使用 SSH URL，不修改所有 worktree 共享的 origin。
git push git@github.com:kkyyxhll/pretrain_exp.git HEAD:alice/qwen35-loader
```

GitHub CLI 使用自己的配置目录，交互登录，不在命令中传 token：

```bash
export GH_CONFIG_DIR="$PERSONAL_STATE_DIR/gh"
mkdir -p "$GH_CONFIG_DIR"
chmod 700 "$GH_CONFIG_DIR"
gh auth login --hostname github.com --git-protocol ssh
gh auth status
```

同一 Linux 账号仍共享操作系统访问权限，worktree 本身只隔离开发目录；需要账号级隔离时使用独立 Linux 用户。

## 安装当前 worktree 的格式检查 hook

pre-commit 和 Ruff 的版本在 `requirements.txt` 中固定，使用已有 olmo 环境。维护者协调安装依赖后，个人 worktree 按以下方式设置：

```bash
conda activate olmo
export PRE_COMMIT_HOME="$PERSONAL_STATE_DIR/pre-commit"
export HOOK_DIR="$PERSONAL_STATE_DIR/hooks/$(basename "$PWD")"
mkdir -p "$HOOK_DIR"
chmod 700 "$HOOK_DIR"
git config --worktree core.hooksPath "$HOOK_DIR"
pre-commit install-hooks
python - <<'PY'
import os
import shlex
import sys
from pathlib import Path

hook = Path(os.environ['HOOK_DIR']) / 'pre-commit'
hook.write_text(
    '#!/bin/sh\nexec '
    + shlex.quote(sys.executable)
    + ' -m pre_commit run --hook-stage pre-commit "$@"\n'
)
hook.chmod(0o700)
PY
git config --show-origin --get core.hooksPath
pre-commit run --all-files
```

`core.hooksPath` 是当前 worktree 的配置，hook 和缓存归个人目录。这里使用 `install-hooks` 准备配置；[pre-commit 的安装实现](https://github.com/pre-commit/pre-commit/blob/main/pre_commit/commands/install_uninstall.py)会在设置 `core.hooksPath` 时拒绝普通 `pre-commit install --install-hooks`，其默认安装路径又是共享 Git common hooks，因此本流程写入个人 hook。不要删除别人的 hook 或取消其配置。

检查使用当前环境中的 `python -m ruff`，只覆盖自有 Python 与配置指定的检查项；vendored OLMo、Megatron 参考原件和官方评估原件被排除。格式工具修改文件时，先检查差异，再重新暂存并提交。项目测试仍需单独运行，格式通过不等于训练或恢复验证通过。

## 修改、检查和提交

```bash
conda activate olmo
python -m pytest tests -q
pre-commit run --all-files
git diff --check
git status --short
git add src/data/olmo_streaming.py tests/test_streaming.py
git diff --cached
git commit -m "Improve resumable compressed streaming"
git push git@github.com:kkyyxhll/pretrain_exp.git HEAD:alice/qwen35-loader
```

每个功能单独提交。共享 conda 环境由维护者管理；开发者不要自行升级环境中的 PyTorch、CUDA、FlashAttention 或 FLA。需要不同依赖时创建自己的环境。数据通过 `$DATA_DIR` 传入，日志、checkpoint、失败和校准证据统一在 `$EXP_DIR/pretrain_exp-runtime` 下，由 `$RUNTIME_DIR` 指向；每次运行使用包含本人名字和 attempt 的独立新目录。固定 tokenizer 资产放在 `recipes/qwen35-08b/tokenizer-qwen35/` 并忽略 Git；通过 `$TOKENIZER_DIR` 指向已有资产或按 README 下载，不修改共享的固定版本。

## 控制磁盘占用

共享机器磁盘空间有限，启动训练前检查 `$RUNTIME_DIR` 所在磁盘的剩余空间，并为完整 checkpoint 保存预留容量。开发过程中只记录必要的配置、简洁日志、结果、成本和恢复校验摘要，不重复复制语料、tokenizer、大型 trace 或其他大文件。

checkpoint 只保留当前恢复所需的最新完整状态，以及仍需验证的最终或候选状态。每档完成训练和验证后归档所需权重，再由运行负责人确认没有运行进程或后续恢复依赖，清理已被替代的 checkpoint。失败、取消和重试的原因与成本记录仍保留；不要清理其他开发者的输出或共享 runtime 根目录。

## 创建 PR、审查和合并 main

```bash
gh pr create --repo kkyyxhll/pretrain_exp --base main --head alice/qwen35-loader \
  --title "Improve resumable compressed streaming" --body-file /path/to/pr-description.txt
```

PR 描述写清楚问题、修改后的行为、验证结果及恢复兼容性；模型或训练策略改动附上配置、seed、FLOPs 与结果证据。邀请另一位开发者 review，通过检查后由维护者合并。合并方式可选 squash；不得把未审查代码直接推送到 main。

同步 main 时只在自己的 worktree 中执行：

```bash
git fetch git@github.com:kkyyxhll/pretrain_exp.git main:refs/remotes/origin/main
git rebase origin/main
```

如果已推送分支且 rebase 改写了提交，确认该分支由自己独占后使用 `--force-with-lease`；不要强推共享 main。

## 训练期间冻结代码

正式训练要求提交后的干净版本，并记录新仓库 commit、配置、seed、环境、数据/tokenizer 哈希和累计计算量。为运行创建独立 detached worktree：

```bash
revision=$(git rev-parse HEAD)
git worktree add --detach "$EXP_DIR/worktrees/run-qwen35-$revision" "$revision"
```

训练启动后不修改这个 worktree 的模型/loader/训练源码。开发在其他 worktree 继续，GPU 使用先协调，不能停止其他人的作业。DP2 完整恢复要求两 rank 的模型、optimizer、trainer、RNG、数据 cursor 和 FLOPs 全部加载且身份不变；不能跨并行度宣称完整恢复。

新 worktree 不会复制忽略的 tokenizer 资产。启动时将 `$TOKENIZER_DIR` 指向已校验的 `recipes/qwen35-08b/tokenizer-qwen35/` 绝对目录，运行会记录其身份；不要因新建 worktree 而下载预训练模型权重。目录迁移或格式修改会改变源码身份，旧工程校准的完整恢复验证不能自动当作新版本的 GPU 验证。

完成并合并后，确认无运行进程依赖自己的开发目录、改动全部提交且证据放在仓库外，再由本人清理：

```bash
cd "$EXP_DIR/pretrain_exp"
git worktree remove "$EXP_DIR/worktrees/alice-qwen35-loader"
git branch -d alice/qwen35-loader
# squash合并时 -d 可能拒绝；先保留分支。本人核对GitHub PR已合并、
# 所需提交与证据已保存后，再自行决定是否使用 -D 删除自己的分支。
git worktree prune
```
