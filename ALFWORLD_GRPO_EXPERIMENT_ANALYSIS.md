# `examples/grpo_trainer/run_alfworld.sh` 实验代码详解

本文以 `examples/grpo_trainer/run_alfworld.sh` 为入口，梳理 ALFWorld + GRPO 实验的完整代码路径。重点是 rollout 期间模型与环境交互的实现，所有关键判断都标注到具体文件和行号。

## 1. 总览

这条实验链路可以概括为：

1. `examples/grpo_trainer/run_alfworld.sh:1-67` 设置实验参数，先生成占位 parquet 数据，再启动 `verl.trainer.main_ppo`。
2. `verl/trainer/main_ppo.py:29-51` 用 Hydra 读取 `ppo_trainer.yaml`，初始化 Ray，然后在 Ray actor `TaskRunner` 中执行训练。
3. `verl/trainer/main_ppo.py:68-78` 下载或定位模型，创建 ALFWorld 训练/验证环境，加载 tokenizer/processor。
4. `verl/trainer/main_ppo.py:145-163` 选择 `EpisodeRewardManager`，构造 `TrajectoryCollector`，这个 collector 是多轮 rollout 的核心。
5. `verl/trainer/main_ppo.py:166-188` 创建数据集、sampler 和 `RayPPOTrainer`，把 `traj_collector/envs/val_envs` 注入 trainer。
6. `verl/trainer/ppo/ray_trainer.py:1007-1295` 进入训练循环；每个 batch 在 `1081-1087` 调 `traj_collector.multi_turn_loop(...)` 完成 agent-environment rollout。
7. `agent_system/multi_turn_rollout/rollout_loop.py:484-539` 按训练/验证选择 rollout 方式，训练时先将每个 prompt 按 `env.rollout.n` 重复成 GRPO group。
8. `agent_system/multi_turn_rollout/rollout_loop.py:306-414` 是真正的环境交互循环：reset 环境、构造 prompt、生成动作、step 环境、累计 reward/done/info、记录轨迹。
9. `agent_system/environments/env_manager.py:133-242` 是 ALFWorld 的 manager，负责把原始 TextWorld observation/admissible actions/history 包装成 LLM 输入，并把 LLM 输出投影成环境动作。
10. `agent_system/environments/env_package/alfworld/envs.py:55-206` 用 Ray remote actor 包装 ALFWorld 子环境，执行并行 `reset/step`。
11. `agent_system/reward_manager/episode.py:29-96` 把每条 trajectory 的 episode reward 放到对应 response 的最后一个有效 token 上。
12. `verl/trainer/ppo/ray_trainer.py:1219-1236` 用 GRPO 计算 advantage，`verl/trainer/ppo/core_algos.py:113-174` 按同一 prompt group 的 reward 均值/方差归一化。

## 2. 启动脚本参数

入口文件：`examples/grpo_trainer/run_alfworld.sh:1-67`。

### Shell 变量

- `ENGINE=${1:-vllm}`：第一个命令行参数指定 rollout engine，默认 `vllm`，随后写入 `actor_rollout_ref.rollout.name=$ENGINE`，见 `run_alfworld.sh:2` 和 `:41`。
- `VLLM_ATTENTION_BACKEND=XFORMERS`：指定 vLLM attention backend，见 `run_alfworld.sh:3`。
- `num_cpus_per_env_worker=0.1`：每个 ALFWorld Ray 环境 worker 申请 0.1 CPU，传到 `env.resources_per_worker.num_cpus`，见 `run_alfworld.sh:5` 和 `:57`。实际 remote actor 创建在 `agent_system/environments/env_package/alfworld/envs.py:101-106`。
- `train_data_size=16`：训练 prompt 个数，见 `run_alfworld.sh:7`。同时用于数据预处理 `--train_data_size` 和训练配置 `data.train_batch_size`。
- `val_data_size=128`：验证 prompt 个数，见 `run_alfworld.sh:8`。同时用于数据预处理 `--val_data_size` 和 `data.val_batch_size`。
- `group_size=8`：每个训练 prompt 采样 8 条环境轨迹，用于 GRPO 分组，见 `run_alfworld.sh:9` 和 `:56`。注意这里使用 `env.rollout.n`，不是 `actor_rollout_ref.rollout.n`；`main_ppo.py:159` 明确要求后者保持为 1。

### 数据准备参数

`run_alfworld.sh:11-15` 调用：

```bash
python3 -m examples.data_preprocess.prepare \
  --mode text \
  --train_data_size 16 \
  --val_data_size 128
```

`examples/data_preprocess/prepare.py:24-31` 定义参数：

- `--mode text`：选择文本模式。`prepare.py:48-51` 中 `text` 模式的 prompt 内容为空字符串，视觉模式才放 `<image>`。
- `--local_dir ~/data/verl-agent/`：默认输出目录，`prepare.py:27`。代码在 `prepare.py:34` 追加 mode，因此脚本实际写到 `~/data/verl-agent/text/`。
- `--hdfs_dir None`：默认不拷贝到 HDFS，见 `prepare.py:28` 和 `:102-104`。
- `--train_data_size/--val_data_size`：分别 select train/test 的前 N 条，见 `prepare.py:45-46`。

重要细节：这个数据集只是占位。`prepare.py:36-43` 加载 `hiyouga/geometry3k`，但注释说明“不使用其中任务内容，只用它表示 modality 和 data size”。在 `text` 模式下，每条数据只保留空 prompt、`data_source=text`、`ability=agent`、`extra_info`，见 `prepare.py:76-88`。真正的任务来自 ALFWorld 环境 reset 后的 observation。

### Hydra 覆盖参数逐项说明

`algorithm.adv_estimator=grpo`：选择 GRPO advantage。训练循环在 `ray_trainer.py:1219-1236` 调 `compute_advantage`，GRPO 分支在 `ray_trainer.py:283-299`，底层计算在 `core_algos.py:113-174`。

`data.train_files=$HOME/data/verl-agent/text/train.parquet` / `data.val_files=...test.parquet`：训练/验证数据路径，传入 `create_rl_dataset`，见 `main_ppo.py:166-167`。

`data.train_batch_size=16`：DataLoader 单个训练 batch 的原始 prompt 数。ALFWorld 训练环境数量是 `train_batch_size * env.rollout.n`，见 `env_manager.py:609-643` 和 `envs.py:98`。

`data.val_batch_size=128`：验证 batch size。验证环境 `group_n=1`，见 `env_manager.py:643`。

`data.max_prompt_length=2048`：每一步环境 observation 被 chat template 包装后，最多 2048 token。tokenize 在 `rollout_loop.py:132-137`，超长处理在 `:161-172`。

`data.max_response_length=512`：每次模型动作生成的最大 response token 数。默认配置将 `actor_rollout_ref.rollout.response_length` 绑定到它，见 `ppo_trainer.yaml:111-112`。

`data.filter_overlong_prompts=True`：RLHFDataset 读取占位数据时会过滤超长 prompt；脚本占位 prompt 为空，实际 rollout 中每步 observation 的长度检查在 `TrajectoryCollector.preprocess_single_sample` 里再次发生，见 `rollout_loop.py:161-172`。

`data.truncation='error'`：prompt 超过 `max_prompt_length` 时直接报错，见 `rollout_loop.py:171-172`。

`data.return_raw_chat=True`：保留原始 chat 结构。初始数据集和每步 observation 处理都会保留 `raw_prompt`；rollout 中在 `rollout_loop.py:185-187` 写回。

`actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct`：actor/ref/rollout 使用的基础模型路径。`main_ppo.py:68` 通过 `copy_to_local` 获取本地路径，`main_ppo.py:77-78` 用它初始化 tokenizer/processor。

`actor_rollout_ref.actor.optim.lr=1e-6`：actor 优化器学习率。默认位置在 `ppo_trainer.yaml:71-79`，实际更新 metrics 中会记录 lr，见 `fsdp_workers.py:624-626`。

`actor_rollout_ref.model.use_remove_padding=True`：启用去 padding 优化。`main_ppo.py:89-94` 选择 FSDP actor worker 后，worker 初始化中会把该配置写入 actor/ref，见 `fsdp_workers.py:560-588`。

`actor_rollout_ref.actor.ppo_mini_batch_size=256`：actor 更新时的 mini-batch 大小。`dp_actor.py:332-340` 按它切分数据。

`actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32`：每张 GPU 的 actor micro-batch。`dp_actor.py:344-356` 按它继续切分 micro-batch 并做梯度累积。

`actor_rollout_ref.actor.use_kl_loss=True`：在 actor loss 中加入 reference KL loss。`dp_actor.py:327-328` 会要求 batch 包含 `ref_log_prob`，`dp_actor.py:418-426` 计算 KL 并乘系数加入 policy loss。

`actor_rollout_ref.actor.kl_loss_coef=0.01`：actor KL loss 权重，见 `dp_actor.py:424-426`。

`actor_rollout_ref.actor.kl_loss_type=low_var_kl`：KL 估计形式，传入 `kl_penalty`，见 `dp_actor.py:421`。

`actor_rollout_ref.model.enable_gradient_checkpointing=True`：启用 gradient checkpointing，默认配置项在 `ppo_trainer.yaml:34`。

`actor_rollout_ref.actor.fsdp_config.param_offload=False` / `optimizer_offload=False`：actor 参数和优化器不 offload 到 CPU。worker update 时只有对应 flag 为真才加载/卸载，见 `fsdp_workers.py:606-609` 和 `:634-639`。

`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32`：重算 rollout old log prob 时的 micro-batch。`fsdp_workers.py:685-689` 写入 `data.meta_info`，`dp_actor.py:274-291` 使用它切分。

`actor_rollout_ref.rollout.tensor_model_parallel_size=2`：vLLM rollout tensor parallel size。默认配置位于 `ppo_trainer.yaml:121`，用于 rollout engine 初始化。

`actor_rollout_ref.rollout.name=$ENGINE`：选择 rollout backend。`main_ppo.py:81-86` 对 `vllm` 做版本检查；实际 worker 的 `generate_sequences` 入口是 `fsdp_workers.py:643-669`。

`actor_rollout_ref.rollout.gpu_memory_utilization=0.6`：vLLM KV cache/显存利用率上限，默认项见 `ppo_trainer.yaml:115`。

`actor_rollout_ref.rollout.enable_chunked_prefill=False`：关闭 vLLM chunked prefill。默认配置和注释见 `ppo_trainer.yaml:130`。

`actor_rollout_ref.rollout.enforce_eager=False`：不强制 eager，允许 vLLM 使用更高性能路径。默认项见 `ppo_trainer.yaml:117`。

`actor_rollout_ref.rollout.free_cache_engine=False`：生成后不释放 vLLM cache engine。释放逻辑见 `vllm_rollout.py:284-286`；若为 true，生成前还会 init cache，见 `vllm_rollout.py:185-187`。

`actor_rollout_ref.rollout.val_kwargs.temperature=0.4` / `do_sample=True`：验证时采样温度和是否采样。验证 meta info 在 `ray_trainer.py:732-738` 设置；vLLM validate 分支在 `vllm_rollout.py:215-222` 使用 `val_kwargs`。

`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32`：reference log prob 的 micro-batch。`fsdp_workers.py:729-733` 写入 meta。

`actor_rollout_ref.ref.fsdp_config.param_offload=True`：reference 模型参数 offload 到 CPU，降低显存。默认配置位置见 `ppo_trainer.yaml:89-99`。

`actor_rollout_ref.actor.use_invalid_action_penalty=True`：启用无效动作惩罚。训练循环在 `ray_trainer.py:1203-1208` 调用，惩罚函数在 `ray_trainer.py:200-224`。

`actor_rollout_ref.actor.invalid_action_penalty_coef=0.1`：每个 invalid action 在该步 response 最后有效 token 上扣 0.1，见 `ray_trainer.py:213-217`。

`algorithm.use_kl_in_reward=False`：不把 KL penalty 加进 reward，只在 actor loss 中使用 KL。对应 reward KL 分支见 `ray_trainer.py:1210-1215`；脚本中为 false，因此 `token_level_rewards=token_level_scores`。

`env.env_name=alfworld/AlfredTWEnv`：选择文本版 ALFWorld。`make_envs` 在 `env_manager.py:630-648` 进入 ALFWorld 分支，并使用 `AlfWorldEnvironmentManager`。

`env.seed=0`：环境随机种子。训练环境使用 seed 0，验证环境使用 seed 1000，见 `env_manager.py:642-643`；每组内 8 条轨迹共享同一个 base seed，见 `envs.py:104-106` 的 `seed + (i // group_n)`。

`env.max_steps=50`：每条 episode 最多环境交互 50 步。rollout for-loop 在 `rollout_loop.py:332`。

`env.rollout.n=8`：训练时每个 prompt 重复 8 次，作为 GRPO group。重复发生在 `rollout_loop.py:503-505`；同组 uid 分配在 `rollout_loop.py:314-323`。

`env.resources_per_worker.num_cpus=0.1`：每个 `AlfworldWorker` Ray actor 资源，传入 `ray.remote(**resources_per_worker)`，见 `envs.py:101-106`。

`trainer.critic_warmup=0`：从第 0 步开始允许 actor 更新。actor 更新条件在 `ray_trainer.py:1245-1250`。

`trainer.logger=['console','wandb']`：日志后端。`Tracking` 初始化见 `ray_trainer.py:1018-1023`。

`trainer.project_name='verl_agent_alfworld'` / `experiment_name='grpo_qwen2.5_1.5b'`：日志项目和实验名，同时影响默认 checkpoint 目录，见 `ppo_trainer.yaml:263-281`。

`trainer.n_gpus_per_node=2` / `trainer.nnodes=1`：资源池 GPU 数。`main_ppo.py:115-122` 用它构造 global resource pool；`adjust_batch` 也用它算 batch size 对齐倍数，见 `utils.py:86-97`。

`trainer.save_freq=-1`：不周期性保存 checkpoint。保存条件见 `ray_trainer.py:1278-1280`。

`trainer.test_freq=5`：每 5 个 global step 验证一次。条件见 `ray_trainer.py:1270-1276`。

`trainer.total_epochs=150`：训练 epoch 数。外层 epoch loop 在 `ray_trainer.py:1047`。

`trainer.val_before_train=True`：训练前先验证。执行位置在 `ray_trainer.py:1030-1037`。

`$@`：允许在命令行继续追加 Hydra override，见 `run_alfworld.sh:67`。

## 3. 配置默认值与脚本覆盖关系

默认配置文件是 `verl/trainer/config/ppo_trainer.yaml`，由 `@hydra.main(config_path="config", config_name="ppo_trainer")` 加载，见 `main_ppo.py:29-31`。

与本实验强相关的默认值：

- `data.*` 默认项在 `ppo_trainer.yaml:1-25`，脚本覆盖了文件路径、batch size、prompt/response 长度、过滤策略和 `return_raw_chat`。
- `actor_rollout_ref.model/actor/ref/rollout` 默认项在 `ppo_trainer.yaml:27-152`。脚本没有显式覆盖 `actor_rollout_ref.rollout.n`，因此保持默认 `1`，满足 `main_ppo.py:159` 的断言。
- `reward_model.reward_manager=episode` 和 `reward_model.enable=False` 在 `ppo_trainer.yaml:202-226`。本实验用环境 reward，不启用模型 reward。
- `algorithm.gamma=1.0`、`lam=1.0`、`norm_adv_by_std_in_grpo=True`、`filter_groups.enable=False` 在 `ppo_trainer.yaml:234-258`。脚本只把 `adv_estimator` 改成 `grpo`，不启用 DAPO dynamic filtering。
- `trainer.*` 默认项在 `ppo_trainer.yaml:259-286`。脚本把 GPU、epoch、test/save/log/project 等覆盖掉。
- `env.*` 默认项在 `ppo_trainer.yaml:292-316`，本实验仍是 `alfworld/AlfredTWEnv`，history 默认 `2`，max steps 被脚本显式设为 `50`。

## 4. main_ppo 启动流程

`verl/trainer/main_ppo.py:34-51` 是进程入口：

- `run_ppo` 检查 Ray 是否已初始化，未初始化时合并默认 runtime env 并 `ray.init`，见 `main_ppo.py:35-48`。
- `TaskRunner.remote()` 创建一个 Ray actor，`ray.get(runner.run.remote(config))` 在 actor 内执行训练，见 `main_ppo.py:50-51`。

`TaskRunner.run` 的关键路径：

- `main_ppo.py:64-65` 打印并 resolve Hydra config。
- `main_ppo.py:68` 把 `actor_rollout_ref.model.path` copy/localize 到本地。
- `main_ppo.py:70-71` 调 `make_envs(config)` 创建训练和验证环境。
- `main_ppo.py:76-78` 创建 tokenizer 和 processor。ALFWorld text 模式下 processor 通常不是核心。
- `main_ppo.py:89-104` 根据 `actor_rollout_ref.actor.strategy` 选择 FSDP 或 Megatron worker。本脚本沿用默认 `fsdp`，所以使用 `ActorRolloutRefWorker`。
- `main_ppo.py:110-122` 把 actor/critic/ref/reward model 映射到同一个 global GPU resource pool。
- `main_ppo.py:141-143` 因为 `actor.use_kl_loss=True`，即使 `algorithm.use_kl_in_reward=False`，也会创建 reference policy。
- `main_ppo.py:145-155` 选择 `EpisodeRewardManager` 作为训练和验证 reward function。
- `main_ppo.py:159` 断言 `actor_rollout_ref.rollout.n == 1`。这个 repo 的 agent 环境 GRPO 不走 vLLM 的 n，而走 `env.rollout.n`。
- `main_ppo.py:161-163` 创建 `TrajectoryCollector`，它负责多步环境交互。
- `main_ppo.py:166-168` 创建 RL dataset 和 sampler。
- `main_ppo.py:169-186` 创建 `RayPPOTrainer`，把 `traj_collector/envs/val_envs` 注入。
- `main_ppo.py:187-188` 初始化 workers 并开始 `fit()`。

## 5. 环境创建路径

入口：`agent_system/environments/env_manager.py:602-648`。

- `env_manager.py:607-609` 检查 `env.rollout.n` 是整数，并设置 `group_n`。本实验 `group_n=8`。
- `env_manager.py:610` 将 `env.resources_per_worker` 转成普通 dict，例如 `{num_cpus: 0.1, num_gpus: 0}`。
- `env_manager.py:630-631` 根据 `env_name` 进入 ALFWorld 分支，导入 `build_alfworld_envs` 和 `alfworld_projection`。
- `env_manager.py:632-635` 对 `alfworld/AlfredTWEnv` 使用 `env_package/alfworld/configs/config_tw.yaml`。
- `env_manager.py:639-641` 读取 `env.alfworld.eval_dataset`，默认 `eval_in_distribution`。
- `env_manager.py:642` 创建训练环境：`env_num=config.data.train_batch_size`，`group_n=8`，因此实际 worker 数为 `16*8=128`。
- `env_manager.py:643` 创建验证环境：`env_num=config.data.val_batch_size`，`group_n=1`，实际 worker 数为 128。
- `env_manager.py:645-647` 用 `AlfWorldEnvironmentManager` 包装底层 env，并把 projection function 注入。

底层 ALFWorld env 在 `agent_system/environments/env_package/alfworld/envs.py`：

- `envs.py:30-34` 读取 YAML 配置。
- `envs.py:85-99` 初始化 `AlfworldEnvs`，从 YAML 读 `env.type`，创建 `base_env = get_environment(env_type)(...)`，并根据是否 `AlfredThorEnv` 判断多模态。
- `envs.py:98` 设置 `self.num_processes = env_num * group_n`。
- `envs.py:101-106` 用 `ray.remote(**resources_per_worker)(AlfworldWorker)` 创建每个 worker；同组 worker 使用相同 seed，代码是 `seed + (i // self.group_n)`。
- `envs.py:108` 用 `prev_admissible_commands` 缓存每个子环境最近一次可行动作列表。
- `envs.py:205-206` 的 `build_alfworld_envs` 只是返回 `AlfworldEnvs(...)`。

单个环境 worker：

- `envs.py:55-63` 的 `AlfworldWorker.__init__` 持有一个 `base_env.init_env(batch_size=1)` 的 TextWorld 环境，并调用 `seed`。
- `envs.py:65-71` 的 `step` 将单个 action 包成 `[action]` 调 TextWorld env，返回 obs/scores/dones/infos，并把 observation_text 填到 info。
- `envs.py:73-77` 的 `reset` 重置单个 TextWorld env，返回 obs/info。

TextWorld ALFWorld 实现：

- `alfred_tw_env.py:112-129` 定义 `AlfredTWEnv`，初始化时收集 game files。
- `alfred_tw_env.py:137-142` 根据 `train/eval_in_distribution/eval_out_of_distribution` 选择数据路径。
- `alfred_tw_env.py:147-193` 遍历数据目录，只收集 task type 符合配置、存在 `game.tw-pddl`、标记 solvable 的游戏。
- `alfred_tw_env.py:198-207` 根据 YAML 中 `num_train_games/num_eval_games` 截断游戏数量。
- `alfred_tw_env.py:245-280` 注册 TextWorld gym env，request infos 包含 `won/admissible_commands/extra.gamefile`，并设置 `max_episode_steps`。

## 6. ALFWorld observation 与 prompt 构造

`AlfWorldEnvironmentManager` 位于 `agent_system/environments/env_manager.py:133-242`。

reset 路径：

- `env_manager.py:138-139` 调底层 `self.envs.reset()`，拿到 `text_obs, image_obs, infos`。
- `env_manager.py:140` 从 info 中解析 `extra.gamefile`。
- `env_manager.py:142-145` 重置 history memory，保存原始 observation，并从 observation 中抽取任务描述。
- `env_manager.py:147-148` 调 `build_text_obs(..., init=True)` 生成 LLM 可读的完整 prompt，返回 `{'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}`。

step 路径：

- `env_manager.py:150-152` 接收模型生成的 `text_actions`，用 `projection_f(text_actions, self.envs.get_admissible_commands)` 得到环境动作和 valid 标记，再调用底层 env step。
- `env_manager.py:153-154` 把上一步 observation 和执行动作写入 memory。
- `env_manager.py:156` 基于新 observation 和 admissible commands 构造下一步 prompt。
- `env_manager.py:157-158` 如果 info 丢了 gamefile，就从 reset 时缓存的 gamefile 补回。
- `env_manager.py:161-163` 把 `is_action_valid` 写进 info，后续用于 invalid action penalty。
- `env_manager.py:164-168` 返回下一个 observation、reward、done、info，reward/done 会转成 numpy。

prompt 模板：

- 无历史模板在 `agent_system/environments/prompts/alfworld.py:17-25`，包含当前 observation 和 admissible actions，要求模型先输出 `<think>...</think>`，再输出 `<action>...</action>`。
- 有历史模板在 `prompts/alfworld.py:27-36`，额外包含任务描述、已执行步数、最近 history。

`build_text_obs` 细节：

- `env_manager.py:180-189` 如果不是初始步且 `env.history_length>0`，从 `SimpleMemory` 中取最近若干步 observation/action。默认 history_length 是 2，见 `ppo_trainer.yaml:296`。
- `env_manager.py:191-193` 过滤 admissible actions 中的 `help`，并把每个 action 格式化成带引号的列表。
- `env_manager.py:195-199` 初始步使用无历史模板。
- `env_manager.py:200-209` 后续步使用有历史模板，包括 task、step_count、history、current_observation 和 admissible actions。

## 7. 动作解析与有效性

动作投影函数：`agent_system/environments/env_package/alfworld/projection.py:19-62`。

输入是 LLM 解码出的 response 字符串列表，以及每个环境的 admissible action pool。当前实现只解析格式和中文字符，不检查 action 是否真的在 `action_pools` 中。

具体逻辑：

- `projection.py:26` 初始化 `valids=[0]*len(actions)`。
- `projection.py:29-30` 保留原字符串用于格式检查，再把 action 转小写。
- `projection.py:32-47` 查找 `<action>...</action>`；找到则抽取标签内文本并标记 `valids[i]=1`，没找到则取 response 最后 30 个字符作为 fallback action。
- `projection.py:52-56` 要求原字符串包含 `<think>` 和 `</think>`，否则 valid 置 0。
- `projection.py:58-60` 如果原字符串包含中文字符，valid 置 0。
- `projection.py:62` 返回 `(actions, valids)`。

这个 `valids` 不会阻止环境 step。即使 invalid，fallback 或抽取出的 action 仍会传给 ALFWorld；惩罚发生在训练 reward 上，见 `ray_trainer.py:200-224`。

## 8. Rollout 过程：环境交互核心

训练循环中触发 rollout 的位置：

- `ray_trainer.py:1047-1051` 遍历训练 DataLoader，并把 batch dict 转成 `DataProto`。
- `ray_trainer.py:1053-1067` 从 batch 中 pop 出生成所需字段：`input_ids/attention_mask/position_ids/raw_prompt_ids/data_source/raw_prompt/env_kwargs` 等，形成 `gen_batch`。
- `ray_trainer.py:1081-1087` 调 `self.traj_collector.multi_turn_loop(gen_batch, actor_rollout_wg, envs, is_train=True)`。

验证 rollout 类似：

- `ray_trainer.py:701-705` 遍历验证 DataLoader，并按 `actor_rollout_ref.rollout.val_kwargs.n` 重复。脚本没有覆盖这个 n，默认是 1，见 `ppo_trainer.yaml:140-146`。
- `ray_trainer.py:732-738` 设置验证 meta info：`recompute_log_prob=False`、`do_sample=val_kwargs.do_sample`、`validate=True`。
- `ray_trainer.py:748-754` 调 `multi_turn_loop(..., envs=self.val_envs, is_train=False)`。

### `multi_turn_loop`

位置：`agent_system/multi_turn_rollout/rollout_loop.py:484-539`。

- `rollout_loop.py:503-505` 如果是训练，先把 `gen_batch` 按 `env.rollout.n` repeat。这里 `16` 个原始 prompt 变成 `128` 条并行轨迹。
- `rollout_loop.py:507-514` 如果 `algorithm.filter_groups.enable=True`，走 dynamic rollout；脚本默认 false。
- `rollout_loop.py:516-522` 本实验走 `vanilla_multi_turn_loop`。
- `rollout_loop.py:523-526` 检查轨迹、reward、length、uid、tool calling 数组长度一致。
- `rollout_loop.py:530-537` 调 `gather_rollout_data` 把按环境收集的 step 数据拼成训练 batch。

### `vanilla_multi_turn_loop`

位置：`agent_system/multi_turn_rollout/rollout_loop.py:285-414`。

初始化：

- `rollout_loop.py:306` batch size 是重复后的轨迹数，训练中为 `train_batch_size * env.rollout.n = 128`。
- `rollout_loop.py:309` 调 `envs.reset(...)` 获取初始 observation 和 info。ALFWorld manager 的 reset 见 `env_manager.py:138-148`。
- `rollout_loop.py:311-312` 确认环境返回的 observation 数量等于 batch size。
- `rollout_loop.py:314-323` 构造 `uid_batch`。训练中每连续 `env.rollout.n=8` 条轨迹共享同一个 uid，作为 GRPO group id。
- `rollout_loop.py:324-330` 初始化 `is_done`、每条轨迹唯一 `traj_uid`、轨迹数据列表、episode length/reward/tool_calling 计数。

每一步环境交互：

- `rollout_loop.py:332` 最多循环 `env.max_steps=50` 步。
- `rollout_loop.py:333` 用 `active_masks = ~is_done` 标记哪些环境仍活跃。
- `rollout_loop.py:335` 调 `preprocess_batch`，把当前 observation 转成模型输入。
- `rollout_loop.py:337-348` 从 batch 中 pop 出生成需要的 tensor/non-tensor 字段，构造 `batch_input`。
- `rollout_loop.py:350` 把原始 `gen_batch.meta_info` 传给本步 input。
- `rollout_loop.py:352-356` 为了能被 worker world size 整除，先 pad `DataProto`，调用 `actor_rollout_wg.generate_sequences`，再 unpad。
- `rollout_loop.py:361` 把生成输出 union 回当前 step 的 batch。
- `rollout_loop.py:363` 解码 `batch.batch['responses']` 得到文本动作。
- `rollout_loop.py:365` 调 `envs.step(text_actions)`，这是模型与环境交互的关键调用。对 ALFWorld 来说，内部会解析 `<action>`、step TextWorld、返回新 observation/reward/done/info。
- `rollout_loop.py:368-372` 如果 reward/done 多一维就 squeeze。
- `rollout_loop.py:374-377` 从 infos 取 `is_action_valid`，没有则默认全 valid。
- `rollout_loop.py:379-380` 如果 info 有 `tool_calling`，累计工具调用次数；ALFWorld 通常没有。
- `rollout_loop.py:383-384` 只对 active 环境累计 reward 和 length。已经 done 的环境虽然仍被批量 step 调用覆盖，但不会继续累计。
- `rollout_loop.py:386-388` 把本步 reward 和 active mask 写入 batch 的 non-tensor 字段。
- `rollout_loop.py:391-395` 将当前 step batch 转为 list of dict，按环境追加到 `total_batch_list/total_infos`。
- `rollout_loop.py:398` 用新 dones 更新 `is_done`。
- `rollout_loop.py:401` 更新 observation，进入下一步。
- `rollout_loop.py:404-405` 如果全部环境 done，提前结束。

收尾：

- `rollout_loop.py:407-412` 调 `envs.success_evaluator` 计算 success metrics。
- `rollout_loop.py:414` 返回所有轨迹、episode reward、episode length、success、traj uid、tool calling。

### observation 到模型输入的转换

位置：`TrajectoryCollector.preprocess_single_sample`，`rollout_loop.py:43-188`。

- `rollout_loop.py:62-64` 从原始 `gen_batch` 取 `raw_prompt`、`data_source` 和 chat template 参数。
- `rollout_loop.py:67-73` 从当前环境 observation 中取 text/image/anchor。ALFWorld TextWorld 只用 text，image 是 None。
- `rollout_loop.py:83-93` 把 observation text 包装成一条 `{"role": "user", "content": obs_content}` chat。
- `rollout_loop.py:96-101` 调 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)`，生成模型实际 prompt。
- `rollout_loop.py:132-137` tokenize 并 left pad 到 `data.max_prompt_length`，truncation 使用脚本设置的 `error`。
- `rollout_loop.py:158-160` 文本模式下用 attention mask 计算 position ids。
- `rollout_loop.py:161-172` 再次检查 raw prompt ids 是否超过 max length，超长按 truncation 策略处理。
- `rollout_loop.py:175-183` 写出 `input_ids/attention_mask/position_ids/raw_prompt_ids/anchor_obs/index/data_source`。
- `rollout_loop.py:185-187` 如果 `return_raw_chat=True`，保留本步 chat。

`preprocess_batch` 在 `rollout_loop.py:190-230` 逐条调用 `preprocess_single_sample`，并用 `collate_fn` 合并成 `DataProto`。

### 模型生成动作

调用链：

- `rollout_loop.py:353-356` 调 worker group 的 `generate_sequences`。
- `verl/workers/fsdp_workers.py:643-669` 是 actor-rollout worker 的入口：把 prompt 移到设备，补 `eos_token_id/pad_token_id`，经 rollout sharding manager preprocess，调用具体 rollout engine。
- vLLM engine 的生成在 `verl/workers/rollout/vllm_rollout/vllm_rollout.py:184-288`。

vLLM 生成细节：

- `vllm_rollout.py:189-203` 取 `input_ids/attention_mask/position_ids`，去掉左 padding，形成 prompt token id list。
- `vllm_rollout.py:204-222` 根据 `do_sample/validate` 决定 sampling 参数。验证时用 `val_kwargs.temperature/top_p/top_k`，且 `n=1`。
- `vllm_rollout.py:232-239` 调 `self.inference_engine.generate(..., prompt_token_ids=idx_list)`。
- `vllm_rollout.py:243-248` 取 response 和 log_probs，不足 `response_length` 则 pad。
- `vllm_rollout.py:256-269` 拼接 prompt+response，并构造 response position ids 和 attention mask。
- `vllm_rollout.py:272-282` 返回包含 `prompts/responses/input_ids/rollout_log_probs/attention_mask/position_ids` 的 `DataProto`。

## 9. Reward、invalid action penalty 与 success

环境 reward 首先来自底层 ALFWorld env：

- `envs.py:127-138` 收集每个 worker step 结果，将 `info` 从 batch 维拆成单条，并用 `compute_reward` 计算 reward。
- `envs.py:48-53` 对 TextWorld 模式 reward 是 `10.0 * float(info['won'])`。也就是说，只有成功完成任务的 step/episode 得到 10，否则通常为 0。

trajectory 级 reward：

- `rollout_loop.py:383` 将每步 reward 累加到 `episode_rewards`。
- `rollout_loop.py:264-277` 在 `gather_rollout_data` 中，只有 `active_masks=True` 的 step 会成为有效训练样本，并写入 `episode_rewards/episode_lengths/tool_callings/success_rate`。

reward manager：

- `EpisodeRewardManager.__call__` 在 `agent_system/reward_manager/episode.py:29-96`。
- `episode.py:39` 创建和 `responses` 同 shape 的零 reward tensor。
- `episode.py:72-79` 读取 `episode_rewards/episode_lengths`，默认不按长度归一化，将 episode score 放到 response 的最后一个有效 token 上。
- 因此每个有效 step 的 response 最后 token 获得整条 episode 的 outcome reward；这使 GRPO 能对同一 prompt group 的轨迹 outcome 做比较。

invalid action penalty：

- `ray_trainer.py:1203-1208` 如果 `actor.use_invalid_action_penalty=True`，调用 `apply_invalid_action_penalty`。
- `ray_trainer.py:213-217` 读取每条样本的 `is_action_valid`，如果 invalid，就在最后有效 response token 上扣 `invalid_action_penalty_coef=0.1`。
- `ray_trainer.py:222-224` 记录 `episode/valid_action_ratio`。

success evaluator：

- 通用 evaluator 在 `agent_system/environments/base.py:114-133`，逐条轨迹调用 `_process_batch`。
- ALFWorld 重写 `_process_batch`，见 `env_manager.py:214-227`：从最后一个 active step 的 info 中读取 `info['won']` 作为 `success_rate`，并根据 `extra.gamefile` 统计不同任务类型的成功率。
- 任务类型解析在 `env_manager.py:229-242`，包括 `pick_and_place`、`pick_two_obj_and_place`、`look_at_obj_in_light`、`pick_heat_then_place_in_recep`、`pick_cool_then_place_in_recep`、`pick_clean_then_place_in_recep`。

## 10. GRPO advantage 与训练更新

训练 batch 形成后：

- `ray_trainer.py:1118` 调 `adjust_batch` 对齐 batch size。
- `agent_system/multi_turn_rollout/utils.py:86-130` 的 `adjust_batch` 根据 rollout/ref/actor micro-batch 和 world size 算最小公倍数，不整除时默认复制若干样本补齐。
- `ray_trainer.py:1120` 计算 response mask。
- `ray_trainer.py:1139` 调 `compute_reward(batch, self.reward_fn)`，最终执行 `EpisodeRewardManager`。
- `ray_trainer.py:1141-1151` 重算 old log prob，入口是 `fsdp_workers.py:672-712`，底层 actor log prob 见 `dp_actor.py:252-314`。
- `ray_trainer.py:1177-1184` 因为 `actor.use_kl_loss=True`，计算 reference log prob。
- `ray_trainer.py:1210-1215` 因为 `algorithm.use_kl_in_reward=False`，不在 reward 中扣 KL，直接 `token_level_rewards=token_level_scores`。
- `ray_trainer.py:1221-1236` 调 `compute_advantage`。

GRPO 计算：

- `ray_trainer.py:283-299` 进入 GRPO 分支，使用 `data.non_tensor_batch["uid"]` 作为 group id，`traj_uid` 作为 trajectory id。
- `core_algos.py:144` 先把 token-level rewards 按 response 求和成每条样本 score。
- `core_algos.py:146-157` 按 `uid` 收集同组分数。默认 `compute_mean_std_cross_steps=True`，因此同一 group 内所有 step 样本都会参与 mean/std，而不是每条 trajectory 只算一次。
- `core_algos.py:158-166` 计算 group mean/std。若组内只有一个分数，则 mean=0/std=1。
- `core_algos.py:167-172` 对每条样本做 `(score - group_mean)/(group_std + epsilon)`，并乘 response mask 扩展到 token 维。
- `core_algos.py:174` returns 和 advantages 都返回这个 score mask。

actor 更新：

- `ray_trainer.py:1245-1250` 满足 critic warmup 后调用 `actor_rollout_wg.update_actor(batch)`。
- `fsdp_workers.py:600-641` 是 worker 入口，转到设备后调用 `self.actor.update_policy`。
- `dp_actor.py:317-329` 选择训练所需字段；因为 `use_kl_loss=True`，包括 `ref_log_prob`。
- `dp_actor.py:332-356` 按 `ppo_mini_batch_size=256` 和 `ppo_micro_batch_size_per_gpu=32` 切分。
- `dp_actor.py:366-388` 取 responses/mask/old_log_probs/advantages，并重新 forward 得到当前 log prob。
- `dp_actor.py:390-408` 默认 `policy_loss.loss_mode=vanilla`，调用 PPO clipped policy loss。
- `dp_actor.py:410-416` 如果 entropy coefficient 非 0，加 entropy bonus。
- `dp_actor.py:418-426` 加 reference KL loss，系数为脚本设置的 `0.01`。
- `dp_actor.py:430-445` 梯度累积、反传、optimizer step，并记录 metrics。

## 11. 验证流程

验证由 `_validate` 实现，见 `verl/trainer/ppo/ray_trainer.py:689-826`。

- `ray_trainer.py:701-705` 读取验证 batch 并按 `val_kwargs.n` repeat。
- `ray_trainer.py:732-738` 设置验证生成参数，`validate=True` 会让 vLLM 使用 `rollout.val_kwargs`。
- `ray_trainer.py:748-754` 调 `TrajectoryCollector.multi_turn_loop(..., is_train=False)`，因此不会按 `env.rollout.n=8` 扩展。
- `ray_trainer.py:765-769` 用 `val_reward_fn` 算分数并收集。
- `ray_trainer.py:775-783` 收集各类 success_rate。
- `ray_trainer.py:787-824` 汇总 reward、tool_call_count、success_rate 等验证指标。

## 12. 需要特别注意的实现细节

1. 本实验的 GRPO group 由 `env.rollout.n` 管理，而不是 `actor_rollout_ref.rollout.n`。`main_ppo.py:159` 已经写死了这个约束。
2. `alfworld_projection` 当前不校验 action 是否属于 admissible action pool，只校验标签格式和中文字符，见 `projection.py:19-62`。因此“格式正确但不在可行动作列表中”的命令会被送进 TextWorld，由环境决定反馈。
3. `EpisodeRewardManager` 把整条 episode reward 写到每个有效 step response 的最后一个 token，见 `episode.py:72-79`；这会让一条成功轨迹中的所有 active step 都带同一个 outcome reward。
4. `compute_grpo_outcome_advantage` 默认跨 step 统计 group mean/std，见 `core_algos.py:120` 和 `:146-157`。这和“每条轨迹一个 outcome 分数再做 GRPO”的实现略有不同。
5. 已 done 的环境仍在批量循环中参与 prompt/generate/step 的形状维护，但 reward/length 只在 `active_masks` 为 true 时累计，见 `rollout_loop.py:333` 和 `:383-388`。
6. 训练环境 worker 数是 `train_batch_size * env.rollout.n`，本脚本是 128 个 Ray actors；验证也是 128 个 actors，但 group_n=1，见 `env_manager.py:642-643` 和 `envs.py:98-106`。
7. 数据 parquet 不是 ALFWorld 任务本身，只负责提供 batch size、modality 和 DataLoader 结构；真实任务在环境 reset 后进入 prompt。

