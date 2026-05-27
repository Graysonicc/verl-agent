# PPO in Agentic RL: 训练流程全面分析

> 基于 `verl-agent` 项目，以 `examples/ppo_trainer/run_alfworld.sh` 为入口，详细分析 PPO 在 Agent-Environment 交互式强化学习中的完整训练流程。

---

## 目录

1. [整体架构概览](#1-整体架构概览)
2. [启动脚本与参数定义](#2-启动脚本与参数定义)
3. [初始化流程](#3-初始化流程)
4. [核心训练循环 (fit)](#4-核心训练循环-fit)
5. [Agent-Environment 多轮交互 (Rollout)](#5-agent-environment-多轮交互-rollout)
6. [奖励计算](#6-奖励计算)
7. [Log Probability 与 Reference Policy 计算](#7-log-probability-与-reference-policy-计算)
8. [Value 估计 (Critic)](#8-value-估计-critic)
9. [优势函数估计 (Advantage Estimation)](#9-优势函数估计-advantage-estimation)
10. [策略更新 (Actor Update)](#10-策略更新-actor-update)
11. [Critic 更新](#11-critic-更新)
12. [数据流向总结](#12-数据流向总结)
13. [完整算法伪代码](#13-完整算法伪代码)
14. [ALFWorld 示例详解](#14-alfworld-示例详解)

---

## 1. 整体架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                     Driver Process (CPU)                        │
│                                                                 │
│  ┌──────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │ DataLoader│──>│ TrajectoryCol│──>│ Advantage Computation │    │
│  └──────────┘   │ lector       │   └──────────────────────┘    │
│                 └──────────────┘                                │
│                        │                                        │
│         ┌──────────────┼──────────────┐                        │
│         ▼              ▼              ▼                        │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                 │
│  │Actor/Rollout│ │   Critic   │ │ Ref Policy │  (Ray Workers) │
│  │  Worker     │ │   Worker   │ │   Worker   │                 │
│  │  (GPU)      │ │   (GPU)    │ │   (GPU)    │                 │
│  └─────┬──────┘ └────────────┘ └────────────┘                 │
│        │                                                        │
│        ▼                                                        │
│  ┌────────────────────────────────────┐                        │
│  │   Environments (Ray Workers/CPU)    │                        │
│  │  ALFWorld / WebShop / Sokoban / ... │                        │
│  └────────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

**核心文件结构：**

| 模块 | 文件路径 | 功能 |
|------|---------|------|
| 入口脚本 | `examples/ppo_trainer/run_alfworld.sh` | 启动训练，配置所有超参数 |
| 主程序 | `verl/trainer/main_ppo.py` | Hydra 入口，初始化各组件 |
| PPO Trainer | `verl/trainer/ppo/ray_trainer.py` | 核心训练循环，编排所有计算步骤 |
| 核心算法 | `verl/trainer/ppo/core_algos.py` | GAE、GRPO、PPO Loss 等算法实现 |
| Actor Worker | `verl/workers/actor/dp_actor.py` | 策略前向/反向、log_prob 计算 |
| Critic Worker | `verl/workers/critic/dp_critic.py` | 价值函数前向/反向 |
| FSDP Worker | `verl/workers/fsdp_workers.py` | 分布式训练 worker 封装 |
| 多轮 Rollout | `agent_system/multi_turn_rollout/rollout_loop.py` | Agent-Env 交互循环 |
| 环境管理 | `agent_system/environments/env_manager.py` | 各环境的封装与管理 |
| 奖励管理 | `agent_system/reward_manager/episode.py` | Episode 级别奖励计算 |
| 默认配置 | `verl/trainer/config/ppo_trainer.yaml` | 所有默认超参数 |

---

## 2. 启动脚本与参数定义

**文件**: `examples/ppo_trainer/run_alfworld.sh`

### 2.1 数据准备

```bash
# run_alfworld.sh:11-14
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \   # 128
    --val_data_size $val_data_size          # 128
```

生成 `train.parquet` 和 `test.parquet`，仅用于指定数据模态和大小（在 agentic RL 中，实际数据来自环境交互）。

### 2.2 完整参数定义

以下是 `run_alfworld.sh` 中设定的关键参数及其含义：

#### 算法参数

| 参数 | 值 | 说明 | 配置文件默认值 |
|------|---|------|--------------|
| `algorithm.adv_estimator` | `gae` | 使用 GAE 作为优势函数估计器 | `gae` (`ppo_trainer.yaml:237`) |
| `algorithm.use_kl_in_reward` | `False` | 不在 reward 中加入 KL 惩罚 | `False` (`ppo_trainer.yaml:239`) |
| `algorithm.gamma` | `1.0` | 折扣因子 | `1.0` (`ppo_trainer.yaml:235`) |
| `algorithm.lam` | `1.0` | GAE 的 lambda 参数 | `1.0` (`ppo_trainer.yaml:236`) |

#### Actor 参数

| 参数 | 值 | 说明 |
|------|---|------|
| `actor_rollout_ref.model.path` | `Qwen/Qwen2.5-1.5B-Instruct` | 策略模型路径 |
| `actor_rollout_ref.actor.optim.lr` | `1e-6` | Actor 学习率 |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | `256` | PPO mini batch 大小 |
| `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu` | `16` | 每 GPU micro batch 大小 |
| `actor_rollout_ref.actor.use_kl_loss` | `True` | 启用 KL loss 正则化 |
| `actor_rollout_ref.actor.kl_loss_coef` | `0.01` | KL loss 系数 |
| `actor_rollout_ref.actor.kl_loss_type` | `low_var_kl` | 低方差 KL 散度估计 |
| `actor_rollout_ref.actor.use_invalid_action_penalty` | `True` | 启用无效动作惩罚 |
| `actor_rollout_ref.actor.invalid_action_penalty_coef` | `0.1` | 无效动作惩罚系数 |
| `actor_rollout_ref.actor.clip_ratio` | `0.2` | PPO clip 范围 (默认) |
| `actor_rollout_ref.actor.entropy_coeff` | `0.001` | 熵正则化系数 (默认) |
| `actor_rollout_ref.actor.ppo_epochs` | `1` | 每步 PPO 更新轮数 (默认) |

#### Critic 参数

| 参数 | 值 | 说明 |
|------|---|------|
| `critic.model.path` | `Qwen/Qwen2.5-1.5B-Instruct` | Critic 模型路径 |
| `critic.optim.lr` | `1e-5` | Critic 学习率 |
| `critic.ppo_micro_batch_size_per_gpu` | `16` | 每 GPU micro batch |
| `critic.cliprange_value` | `0.5` | Value function clip 范围 (默认) |

#### Rollout 参数

| 参数 | 值 | 说明 |
|------|---|------|
| `actor_rollout_ref.rollout.name` | `vllm` | 使用 vLLM 推理引擎 |
| `actor_rollout_ref.rollout.tensor_model_parallel_size` | `2` | 张量并行大小 |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | `0.6` | GPU 显存利用率 |
| `actor_rollout_ref.rollout.temperature` | `1.0` | 采样温度 (默认) |
| `actor_rollout_ref.rollout.val_kwargs.temperature` | `0.4` | 验证时采样温度 |

#### 环境参数

| 参数 | 值 | 说明 |
|------|---|------|
| `env.env_name` | `alfworld/AlfredTWEnv` | ALFWorld 文本环境 |
| `env.seed` | `0` | 随机种子 |
| `env.max_steps` | `50` | 每 episode 最大交互步数 |
| `env.history_length` | `2` | 历史记录长度 (默认) |

#### 训练参数

| 参数 | 值 | 说明 |
|------|---|------|
| `trainer.total_epochs` | `150` | 总训练轮数 |
| `trainer.n_gpus_per_node` | `2` | 每节点 GPU 数 |
| `trainer.nnodes` | `1` | 节点数 |
| `trainer.test_freq` | `5` | 每 5 步做一次验证 |
| `trainer.val_before_train` | `True` | 训练前先验证 |
| `trainer.critic_warmup` | `0` | Critic 预热步数 |
| `data.train_batch_size` | `128` | 训练 batch 大小 |
| `data.max_prompt_length` | `2048` | 最大 prompt 长度 |
| `data.max_response_length` | `512` | 最大 response 长度 |

---

## 3. 初始化流程

### 3.1 程序入口

**文件**: `verl/trainer/main_ppo.py:29-31`

```python
@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    run_ppo(config)
```

使用 Hydra 框架加载 `ppo_trainer.yaml` 默认配置，并用命令行参数覆盖。

### 3.2 Ray 初始化与 TaskRunner

**文件**: `verl/trainer/main_ppo.py:34-51`

```python
def run_ppo(config) -> None:
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()  # line 41
        ray.init(**OmegaConf.to_container(ray_init_kwargs))  # line 48
    runner = TaskRunner.remote()  # line 50
    ray.get(runner.run.remote(config))  # line 51
```

`TaskRunner` 是一个 Ray remote actor，在独立进程中执行训练逻辑。

### 3.3 组件初始化（在 TaskRunner.run 中）

**文件**: `verl/trainer/main_ppo.py:56-188`

依次初始化以下组件：

1. **模型下载** (line 68):
   ```python
   local_path = copy_to_local(config.actor_rollout_ref.model.path, ...)
   ```

2. **环境创建** (line 71):
   ```python
   envs, val_envs = make_envs(config)
   ```
   - 对于 ALFWorld，调用 `agent_system/environments/env_manager.py:630-648`
   - 创建 `AlfWorldEnvironmentManager` 封装训练和验证环境

3. **Tokenizer 创建** (line 77):
   ```python
   tokenizer = hf_tokenizer(local_path, ...)
   ```

4. **Worker 类选择** (line 89-106):
   ```python
   if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
       from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
       # ...
   ```

5. **角色-Worker 映射** (line 110-143):
   ```python
   role_worker_mapping = {
       Role.ActorRollout: ray.remote(actor_rollout_cls),
       Role.Critic: ray.remote(CriticWorker),
   }
   # 如果启用 KL loss，还会创建 RefPolicy:
   if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
       role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
   ```

6. **奖励管理器** (line 146-155):
   ```python
   from agent_system.reward_manager import EpisodeRewardManager
   reward_fn = EpisodeRewardManager(tokenizer=tokenizer, num_examine=0, normalize_by_length=False)
   ```

7. **轨迹收集器** (line 162):
   ```python
   traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
   ```

8. **创建 Trainer 并启动** (line 169-188):
   ```python
   trainer = RayPPOTrainer(config=config, tokenizer=tokenizer, ...)
   trainer.init_workers()  # line 187 → 初始化 Ray Worker Groups
   trainer.fit()           # line 188 → 开始训练循环
   ```

### 3.4 Worker 初始化

**文件**: `verl/trainer/ppo/ray_trainer.py:828-909`

```python
def init_workers(self):
    self.resource_pool_manager.create_resource_pool()   # line 835
    # 创建 colocated worker (Actor+Rollout 共享 GPU)
    worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)  # line 881
    # ...
    self.actor_rollout_wg = all_wg["actor_rollout"]    # line 899
    self.actor_rollout_wg.init_model()                  # line 900
```

所有角色（Actor、Rollout、Critic、Ref）共享同一个 GPU 池 (`global_pool`)，通过 colocated worker 实现权重共享。

---

## 4. 核心训练循环 (fit)

**文件**: `verl/trainer/ppo/ray_trainer.py:1007-1304`

```python
def fit(self):
    # 1. 加载 checkpoint (如果有)
    self._load_checkpoint()                             # line 1028

    # 2. 训练前验证
    if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()                  # line 1033

    # 3. 主循环
    for epoch in range(self.config.trainer.total_epochs):       # line 1047
        for batch_dict in self.train_dataloader:                # line 1048
            batch: DataProto = DataProto.from_single_dict(batch_dict)  # line 1051

            # ===== Step A: 生成轨迹 (Rollout) =====
            gen_batch_output = self.traj_collector.multi_turn_loop(...)  # line 1082-1087

            # ===== Step B: 奖励计算 =====
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)  # line 1139

            # ===== Step C: 计算 old_log_prob =====
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # line 1143

            # ===== Step D: 计算 ref_log_prob =====
            if self.use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # line 1181

            # ===== Step E: 计算 values (Critic) =====
            if self.use_critic:
                values = self.critic_wg.compute_values(batch)   # line 1189

            # ===== Step F: 计算 advantage =====
            batch = compute_advantage(batch, adv_estimator=...) # line 1221

            # ===== Step G: 更新 Critic =====
            if self.use_critic:
                critic_output = self.critic_wg.update_critic(batch)  # line 1241

            # ===== Step H: 更新 Actor =====
            if self.config.trainer.critic_warmup <= self.global_steps:
                actor_output = self.actor_rollout_wg.update_actor(batch)  # line 1250

            # ===== Step I: 验证与保存 =====
            # ...
```

### 每步训练的数据流向：

```
DataLoader → gen_batch
     │
     ▼
[Agent-Env Loop] → gen_batch_output (含 prompts, responses, attention_mask, ...)
     │
     ├──→ [Reward Fn] → token_level_scores
     ├──→ [Actor]     → old_log_probs, entropys
     ├──→ [Ref Policy]→ ref_log_prob
     ├──→ [Critic]    → values
     │
     ▼
[Advantage Estimation] → advantages, returns
     │
     ├──→ [Critic Update] → value loss → grad update
     └──→ [Actor Update]  → policy loss + KL loss → grad update
```

---

## 5. Agent-Environment 多轮交互 (Rollout)

这是 Agentic RL 与传统 RLHF 最大的区别：模型需要与环境进行**多轮交互**，而非一次性生成完整 response。

### 5.1 入口：multi_turn_loop

**文件**: `agent_system/multi_turn_rollout/rollout_loop.py:484-539`

```python
def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
    if is_train:
        gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)  # line 504

    if self.config.algorithm.filter_groups.enable and is_train:
        # 动态采样 (DAPO 风格)
        ... = self.dynamic_multi_turn_loop(...)  # line 509-513
    else:
        # 标准采样
        ... = self.vanilla_multi_turn_loop(...)  # line 517-522

    # 收集轨迹数据
    gen_batch_output = self.gather_rollout_data(...)  # line 530-537
    return gen_batch_output
```

### 5.2 核心：vanilla_multi_turn_loop

**文件**: `agent_system/multi_turn_rollout/rollout_loop.py:285-414`

这是 Agent-Environment 交互的核心循环：

```python
def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
    batch_size = len(gen_batch.batch)

    # Step 1: 环境重置
    obs, infos = envs.reset(...)                                # line 309

    is_done = np.zeros(batch_size, dtype=bool)
    traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
    total_batch_list = [[] for _ in range(batch_size)]          # line 326

    # Step 2: 多轮交互循环 (最多 max_steps 步)
    for _step in range(self.config.env.max_steps):              # line 332
        # 2a. 将环境观测预处理为模型输入
        batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)  # line 335

        # 2b. 模型生成动作 (通过 vLLM)
        batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)  # line 353
        batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)  # line 354
        batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)          # line 356

        # 2c. 解码模型输出为文本动作
        text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)  # line 363

        # 2d. 环境执行动作
        next_obs, rewards, dones, infos = envs.step(text_actions)  # line 365

        # 2e. 记录轨迹数据
        episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]  # line 383
        episode_lengths[active_masks] += 1                                       # line 384

        batch_list = to_list_of_dict(batch)                                      # line 391
        for i in range(batch_size):
            total_batch_list[i].append(batch_list[i])                            # line 394

        # 2f. 更新 done 状态
        is_done = np.logical_or(is_done, dones)                                  # line 398
        obs = next_obs                                                            # line 401

        if is_done.all():
            break                                                                 # line 404-405

    return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
```

### 5.3 单步观测预处理

**文件**: `agent_system/multi_turn_rollout/rollout_loop.py:43-188`

```python
def preprocess_single_sample(self, item, gen_batch, obs):
    obs_text = obs['text'][item]                               # line 71

    # 构建 chat 格式
    chat = np.array([{"content": obs_content, "role": "user"}])  # line 90-93

    # 应用 chat template
    prompt_with_chat_template = self.tokenizer.apply_chat_template(  # line 96-101
        chat, add_generation_prompt=True, tokenize=False, ...
    )

    # Tokenize + padding
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(  # line 132-137
        prompt=prompt_with_chat_template,
        tokenizer=self.tokenizer,
        max_length=self.config.data.max_prompt_length,
        pad_token_id=self.tokenizer.pad_token_id,
        left_pad=True, ...
    )
```

### 5.4 环境管理器 (以 ALFWorld 为例)

**文件**: `agent_system/environments/env_manager.py:133-242`

```python
class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()   # line 139
        self.memory.reset(batch_size=len(text_obs))       # line 142
        self.extract_task(text_obs)                        # line 145
        full_text_obs = self.build_text_obs(text_obs, ..., init=True)  # line 147
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions, ...)  # line 151 (文本→环境动作)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)  # line 152
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})  # line 153
        # ...
```

关键点：`projection_f`（投影函数）将模型生成的自由文本转换为环境可执行的动作，同时返回 `valids` 标记动作是否有效。

### 5.5 轨迹数据收集

**文件**: `agent_system/multi_turn_rollout/rollout_loop.py:233-283`

```python
def gather_rollout_data(self, total_batch_list, episode_rewards, ...):
    effective_batch = []
    for bs in range(batch_size):
        for data in total_batch_list[bs]:
            if data['active_masks']:  # 只保留活跃步的数据
                data['episode_rewards'] = episode_rewards[bs]   # line 268
                data['episode_lengths'] = episode_lengths[bs]   # line 270
                data['tool_callings'] = tool_callings[bs]       # line 272
                effective_batch.append(data)                     # line 277

    gen_batch_output = DataProto.from_single_dict(
        data=collate_fn(effective_batch)                         # line 280-282
    )
```

每个 episode 的每一步都成为一个独立的训练样本。一个 `batch_size=128`、平均 `10` 步的训练 batch 会产生约 `1280` 个训练样本。

---

## 6. 奖励计算

### 6.1 Episode Reward Manager

**文件**: `agent_system/reward_manager/episode.py:20-96`

```python
class EpisodeRewardManager:
    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)  # line 39

        for i in range(len(data)):
            data_item = data[i]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()  # line 54

            episode_rewards = data_item.non_tensor_batch['episode_rewards']   # line 72
            episode_lengths = data_item.non_tensor_batch['episode_lengths']   # line 73
            score = episode_rewards  # (如果不 normalize)                     # line 77

            # 奖励放在 response 最后一个有效 token 的位置
            reward_tensor[i, valid_response_length - 1] = torch.tensor(score)  # line 79
```

关键设计：
- 奖励是 **episode 级别**的，不是 token 级别的
- 每个 step 的 `episode_rewards` 都是该 episode 的总奖励
- 奖励放置在每个 step 的 response 最后一个 token 位置（稀疏奖励）

### 6.2 无效动作惩罚

**文件**: `verl/trainer/ppo/ray_trainer.py:200-224`

```python
def apply_invalid_action_penalty(data, invalid_action_penalty_coef):
    for i in range(len(data)):
        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, ...)                 # line 214
        reward_tensor[i, valid_response_length - 1] -= invalid_action_penalty_coef * action_invalids  # line 217
```

当 Agent 生成无效动作时（如不在 admissible actions 中），额外扣除 `0.1` 的惩罚。

### 6.3 KL 惩罚 (可选)

**文件**: `verl/trainer/ppo/ray_trainer.py:152-198`

```python
def apply_kl_penalty(data, kl_ctrl, kl_penalty="kl", multi_turn=False):
    # 计算当前策略与参考策略的 KL 散度
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], ...)  # line 183
    beta = kl_ctrl.value                                        # line 185
    token_level_rewards = token_level_scores - beta * kld       # line 187
```

在 ALFWorld 配置中，`use_kl_in_reward=False`，所以 KL 惩罚不加入到 reward 中，但 **KL loss 是启用的** (`use_kl_loss=True`)，作为 Actor loss 的正则项。

---

## 7. Log Probability 与 Reference Policy 计算

### 7.1 Old Log Prob 计算

**文件**: `verl/trainer/ppo/ray_trainer.py:1142-1151`

在 Trainer 中调用：
```python
with _timer("old_log_prob", timing_raw):
    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)  # line 1143
    entropys = old_log_prob.batch["entropys"]                      # line 1144
```

实际计算位于 `verl/workers/actor/dp_actor.py:252-314`:

```python
def compute_log_prob(self, data: DataProto, calculate_entropy=False):
    self.actor_module.eval()                                       # line 272
    # ...
    for micro_batch in micro_batches:
        with torch.no_grad():                                      # line 298
            entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
    # ...
    return log_probs, entropys
```

`_forward_micro_batch` (line 76-232) 执行模型前向传播，得到 logits 后计算 log_prob：
```python
# dp_actor.py:228
log_probs = logprobs_from_logits(logits, micro_batch["responses"])
```

### 7.2 Reference Policy Log Prob

**文件**: `verl/trainer/ppo/ray_trainer.py:1177-1184`

```python
if self.use_reference_policy:
    with _timer("ref", timing_raw):
        if not self.ref_in_actor:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)  # line 1181
        else:
            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)  # line 1183
        batch = batch.union(ref_log_prob)                                      # line 1184
```

Reference Policy 使用与 Actor 相同架构的模型，但参数冻结（不更新），用于计算 KL 散度。

---

## 8. Value 估计 (Critic)

### 8.1 计算 Values

**文件**: `verl/workers/critic/dp_critic.py:138-177`

```python
def compute_values(self, data: DataProto):
    self.critic_module.eval()                                     # line 140
    # ...
    for micro_batch in micro_batches:
        with torch.no_grad():
            values = self._forward_micro_batch(micro_batch)       # line 164
    # ...
    values = values * attention_mask[:, -response_length - 1 : -1]  # line 176
    return values
```

Critic 模型的输出 shape 为 `(batch_size, response_length)`，每个 token 位置给出一个 value 估计。

### 8.2 _forward_micro_batch

**文件**: `verl/workers/critic/dp_critic.py:60-118`

```python
def _forward_micro_batch(self, micro_batch):
    output = self.critic_module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    values = output.logits                              # line 98 (Critic 输出 1 维)
    values = values[:, -response_length - 1 : -1]       # 取 response 部分
    return values
```

---

## 9. 优势函数估计 (Advantage Estimation)

### 9.1 GAE (Generalized Advantage Estimation)

在 ALFWorld 配置中使用 GAE (`algorithm.adv_estimator=gae`)。

**入口**: `verl/trainer/ppo/ray_trainer.py:1221-1236`

```python
batch = compute_advantage(
    batch,
    adv_estimator=self.config.algorithm.adv_estimator,   # 'gae'
    gamma=self.config.algorithm.gamma,                    # 1.0
    lam=self.config.algorithm.lam,                        # 1.0
    ...
)
```

**分发**: `verl/trainer/ppo/ray_trainer.py:267-276`

```python
if adv_estimator == AdvantageEstimator.GAE:
    advantages, returns = core_algos.compute_gae_advantage_return(
        token_level_rewards=data.batch["token_level_rewards"],
        values=data.batch["values"],
        response_mask=data.batch["response_mask"],
        gamma=gamma,
        lam=lam,
    )
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
```

### 9.2 GAE 算法实现

**文件**: `verl/trainer/ppo/core_algos.py:67-109`

```python
def compute_gae_advantage_return(token_level_rewards, values, response_mask, gamma, lam):
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):                    # line 100
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0  # line 101
            # δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]  # line 102
            # A_t = δ_t + γ * λ * A_{t+1}
            lastgaelam = delta + gamma * lam * lastgaelam     # line 103
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)  # line 105

        returns = advantages + values                          # line 107
        advantages = verl_F.masked_whiten(advantages, response_mask)  # line 108 (白化标准化)
    return advantages, returns
```

**GAE 数学公式：**

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{T-t} (\gamma\lambda)^l \delta_{t+l}$$
$$\text{Returns}_t = \hat{A}_t + V(s_t)$$

其中：
- `gamma=1.0`: 不折扣未来奖励
- `lam=1.0`: 等价于 Monte Carlo 估计
- 最后对 advantages 做白化处理 (`masked_whiten`)

### 9.3 其他优势估计器

项目还支持以下估计器 (`verl/trainer/ppo/ray_trainer.py:84-96`)：

| 估计器 | 说明 | 需要 Critic |
|--------|------|------------|
| `GAE` | 广义优势估计 | 是 |
| `GRPO` | Group Relative Policy Optimization | 否 |
| `REINFORCE_PLUS_PLUS` | REINFORCE++ | 否 |
| `REINFORCE_PLUS_PLUS_BASELINE` | REINFORCE++ with Baseline | 否 |
| `REMAX` | ReMax | 否 |
| `RLOO` | Leave-One-Out | 否 |
| `GiGPO` | Group-in-Group Policy Optimization | 否 |

---

## 10. 策略更新 (Actor Update)

### 10.1 Trainer 调用

**文件**: `verl/trainer/ppo/ray_trainer.py:1246-1252`

```python
if self.config.trainer.critic_warmup <= self.global_steps:
    with _timer("update_actor", timing_raw):
        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
        actor_output = self.actor_rollout_wg.update_actor(batch)  # line 1250
```

### 10.2 策略更新实现

**文件**: `verl/workers/actor/dp_actor.py:316-447`

```python
def update_policy(self, data: DataProto):
    self.actor_module.train()                                  # line 319
    temperature = data.meta_info["temperature"]                # line 321

    for epoch in range(self.config.ppo_epochs):                # line 342 (默认 1)
        for batch_idx, data in enumerate(dataloader):          # line 343
            self.actor_optimizer.zero_grad()                   # line 358

            for data in micro_batches:                         # line 360
                # 前向传播得到当前策略的 log_prob
                entropy, log_prob = self._forward_micro_batch(data, temperature=temperature)  # line 388

                # === 计算 PPO Clipped Loss ===
                pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    response_mask=response_mask,
                    cliprange=clip_ratio,          # 0.2
                    ...
                )                                              # line 398-408

                # 熵正则化
                if entropy_coeff != 0:
                    entropy_loss = agg_loss(entropy, response_mask, loss_agg_mode)
                    policy_loss = pg_loss - entropy_loss * entropy_coeff  # line 414

                # KL Loss 正则化 (ALFWorld 中启用)
                if self.config.use_kl_loss:
                    kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob,
                                    kl_penalty=self.config.kl_loss_type)  # line 421
                    kl_loss = agg_loss(kld, response_mask, loss_agg_mode)  # line 422
                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef  # line 424
                    # kl_loss_coef = 0.01

                # 梯度累积
                loss = policy_loss / self.gradient_accumulation  # line 433
                loss.backward()                                   # line 434

            # 梯度裁剪 + 优化器步进
            grad_norm = self._optimizer_step()                    # line 443
```

### 10.3 PPO Clipped Policy Loss

**文件**: `verl/trainer/ppo/core_algos.py:431-492`

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, response_mask,
                        cliprange, cliprange_low, cliprange_high, clip_ratio_c, loss_agg_mode):

    negative_approx_kl = log_prob - old_log_prob                    # line 472
    ratio = torch.exp(negative_approx_kl)                           # line 473
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask) # line 474

    # 标准 PPO 目标
    pg_losses1 = -advantages * ratio                                # line 476
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # line 481
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)         # line 482

    # Dual-clip PPO (下界保护)
    pg_losses3 = -advantages * clip_ratio_c                         # line 485 (c=3.0)
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)        # line 486

    # 最终损失: 负优势用 dual-clip, 正优势用标准 clip
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)  # line 489
    pg_loss = agg_loss(pg_losses, response_mask, loss_agg_mode)     # line 490

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower
```

**PPO Clipped Loss 数学公式：**

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

$$L^{CLIP}(\theta) = -\mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Dual-clip 扩展 (当 $\hat{A}_t < 0$ 时):
$$L^{DualCLIP}(\theta) = -\mathbb{E}_t \left[ \max\left( L^{CLIP}_t, c \cdot \hat{A}_t \right) \right]$$

### 10.4 KL 散度类型 (low_var_kl)

**文件**: `verl/trainer/ppo/core_algos.py:638-642`

```python
if kl_penalty in ("low_var_kl", "k3"):
    kl = ref_logprob - logprob
    ratio = torch.exp(kl)
    kld = (ratio - kl - 1).contiguous()  # Schulman 低方差 KL 近似
    return torch.clamp(kld, min=-10, max=10)
```

这是 John Schulman 提出的低方差 KL 近似: $\hat{D}_{KL} = e^{r} - r - 1$，其中 $r = \log \pi_{ref} - \log \pi_\theta$。

### 10.5 Actor 完整 Loss

$$L_{actor} = L^{CLIP}_{PPO} - \alpha_{ent} \cdot H[\pi_\theta] + \alpha_{KL} \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

在 ALFWorld 中: $\alpha_{ent} = 0.001$, $\alpha_{KL} = 0.01$

---

## 11. Critic 更新

**文件**: `verl/workers/critic/dp_critic.py:179-260`

```python
def update_critic(self, data: DataProto):
    self.critic_module.train()                                    # line 182

    for epoch in range(self.config.ppo_epochs):                   # line 198
        for batch_idx, data in enumerate(dataloader):             # line 199
            self.critic_optimizer.zero_grad()                     # line 212

            for data in micro_batches:                            # line 214
                vpreds = self._forward_micro_batch(data)          # line 228

                # Clipped Value Loss
                vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    vpreds=vpreds,
                    values=values,           # 旧的 value 估计
                    returns=returns,          # GAE 计算的 returns
                    response_mask=response_mask,
                    cliprange_value=self.config.cliprange_value,  # 0.5
                    ...
                )                                                  # line 232-239

                loss = vf_loss / self.gradient_accumulation       # line 244
                loss.backward()                                    # line 246

            grad_norm = self._optimizer_step()                     # line 256
```

### Clipped Value Loss

**文件**: `verl/trainer/ppo/core_algos.py:580-612`

```python
def compute_value_loss(vpreds, returns, values, response_mask, cliprange_value, loss_agg_mode):
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)  # line 606
    vf_losses1 = (vpreds - returns) ** 2                        # line 607
    vf_losses2 = (vpredclipped - returns) ** 2                  # line 608
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)       # line 609
    vf_loss = agg_loss(clipped_vf_losses, response_mask, loss_agg_mode)  # line 610
```

$$L^{VF}(\phi) = \mathbb{E}_t \left[ \max \left( (V_\phi(s_t) - R_t)^2, (\text{clip}(V_\phi(s_t), V_{\phi_{old}}(s_t) \pm c_v) - R_t)^2 \right) \right]$$

---

## 12. 数据流向总结

### 12.1 完整数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           Training Step                                 │
│                                                                          │
│  DataLoader                                                              │
│  └──> batch_dict (DataProto)                                            │
│       ├── batch: {input_ids, attention_mask, position_ids}               │
│       └── non_tensor_batch: {raw_prompt, data_source, env_kwargs}       │
│                                                                          │
│  ┌─── Agent-Environment Loop (max_steps=50) ────────────────────┐       │
│  │                                                               │       │
│  │  每一步:                                                      │       │
│  │  obs (text/image) → preprocess → model input                 │       │
│  │  model input → actor_rollout_wg.generate_sequences           │       │
│  │            → responses (token ids)                            │       │
│  │  responses → tokenizer.batch_decode → text_actions           │       │
│  │  text_actions → envs.step → (next_obs, rewards, dones)       │       │
│  │                                                               │       │
│  │  记录: {prompts, responses, attention_mask, rewards,          │       │
│  │         active_masks, uid, traj_uid, is_action_valid,         │       │
│  │         episode_rewards, episode_lengths, anchor_obs}         │       │
│  │                                                               │       │
│  └───────────────────────────────────────────────────────────────┘       │
│       │                                                                  │
│       ▼                                                                  │
│  gen_batch_output (DataProto)                                           │
│  ├── batch: {input_ids, attention_mask, position_ids,                   │
│  │           prompts, responses}                                        │
│  └── non_tensor_batch: {uid, traj_uid, episode_rewards,                │
│                         episode_lengths, is_action_valid,               │
│                         anchor_obs, tool_callings, ...}                 │
│                                                                          │
│  ===== 后续计算 (全部在 batch 上操作) =====                              │
│                                                                          │
│  1. reward_fn(batch)                                                    │
│     └──> batch.batch["token_level_scores"]    # (bs, resp_len)          │
│                                                                          │
│  2. apply_invalid_action_penalty(batch)                                 │
│     └──> 修改 batch.batch["token_level_scores"]                         │
│                                                                          │
│  3. token_level_rewards = token_level_scores (无 KL reward 时)          │
│     └──> batch.batch["token_level_rewards"]                             │
│                                                                          │
│  4. actor_rollout_wg.compute_log_prob(batch)                            │
│     └──> batch.batch["old_log_probs"]         # (bs, resp_len)          │
│                                                                          │
│  5. ref_policy_wg.compute_ref_log_prob(batch)                           │
│     └──> batch.batch["ref_log_prob"]          # (bs, resp_len)          │
│                                                                          │
│  6. critic_wg.compute_values(batch)                                     │
│     └──> batch.batch["values"]                # (bs, resp_len)          │
│                                                                          │
│  7. compute_advantage(batch)  [GAE]                                     │
│     └──> batch.batch["advantages"]            # (bs, resp_len)          │
│     └──> batch.batch["returns"]               # (bs, resp_len)          │
│                                                                          │
│  8. critic_wg.update_critic(batch)                                      │
│     └──> value loss → backward → optimizer step                         │
│                                                                          │
│  9. actor_rollout_wg.update_actor(batch)                                │
│     └──> policy loss + KL loss → backward → optimizer step              │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 12.2 关键张量形状

| 张量 | 形状 | 说明 |
|------|------|------|
| `input_ids` | `(bs, prompt_len + resp_len)` | 完整输入 |
| `attention_mask` | `(bs, prompt_len + resp_len)` | 注意力掩码 |
| `prompts` | `(bs, prompt_len)` | 提示部分 |
| `responses` | `(bs, resp_len)` | 生成部分 |
| `old_log_probs` | `(bs, resp_len)` | 旧策略 log 概率 |
| `ref_log_prob` | `(bs, resp_len)` | 参考策略 log 概率 |
| `values` | `(bs, resp_len)` | Critic 价值估计 |
| `token_level_scores` | `(bs, resp_len)` | token 级奖励（稀疏） |
| `token_level_rewards` | `(bs, resp_len)` | 最终 token 级奖励 |
| `advantages` | `(bs, resp_len)` | 优势估计 |
| `returns` | `(bs, resp_len)` | 回报估计 |

---

## 13. 完整算法伪代码

```
算法: PPO for Agentic RL (以 ALFWorld 为例)

输入:
  - 策略模型 π_θ (Actor, Qwen2.5-1.5B-Instruct)
  - 价值模型 V_φ (Critic, Qwen2.5-1.5B-Instruct)
  - 参考策略 π_ref (冻结参数，用于 KL 正则化)
  - 环境集合 Envs (ALFWorld × 128)
  - 超参数: γ=1.0, λ=1.0, ε=0.2, α_ent=0.001, α_KL=0.01

For epoch = 1 to 150:
  For each training batch:

    ========== 阶段 1: 轨迹收集 ==========

    1. 从 DataLoader 获取 batch (128 个初始提示)
    2. 环境重置: obs, infos ← Envs.reset()
    3. For step = 1 to max_steps (50):
       a. 将 obs 编码为模型输入:
          - 构建 chat 格式 → apply_chat_template → tokenize
       b. 模型生成动作: responses ← π_θ.generate(input)  [via vLLM]
       c. 解码文本动作: text_actions ← decode(responses)
       d. 投影到环境动作: env_actions ← projection_f(text_actions)
       e. 环境执行: next_obs, rewards, dones ← Envs.step(env_actions)
       f. 记录轨迹数据: {input, response, reward, done, is_valid_action, ...}
       g. 若所有环境结束，退出循环
    4. 收集有效轨迹 → effective_batch (~1280 条记录)

    ========== 阶段 2: 评估阶段 ==========

    5. 计算 episode 奖励:
       reward_tensor[i, last_valid_token] = episode_reward_i
    6. 无效动作惩罚:
       reward_tensor[i, last_valid_token] -= 0.1 * (1 - is_valid)
    7. token_level_rewards = token_level_scores  (因 use_kl_in_reward=False)
    8. 计算旧策略 log 概率:
       old_log_prob ← π_θ.forward(batch)  [不计算梯度]
    9. 计算参考策略 log 概率:
       ref_log_prob ← π_ref.forward(batch)  [不计算梯度]
    10. 计算 Critic 价值:
        values ← V_φ.forward(batch)  [不计算梯度]

    ========== 阶段 3: 优势估计 ==========

    11. 计算 GAE:
        For t = T-1 down to 0:
          δ_t = r_t + γ·V(s_{t+1}) - V(s_t)
          A_t = δ_t + γ·λ·A_{t+1}
        returns = A + V
        advantages = whiten(A)  [按 response_mask 白化]

    ========== 阶段 4: 模型更新 ==========

    12. 更新 Critic:
        For each mini-batch, micro-batch:
          V_new = V_φ.forward(micro_batch)
          V_clip = clip(V_new, V_old ± 0.5)
          L_vf = max((V_new - returns)², (V_clip - returns)²)
          L_vf.backward()
        clip_grad_norm(1.0)
        optimizer_φ.step()

    13. 更新 Actor (if step > critic_warmup):
        For each mini-batch, micro-batch:
          log_prob_new = π_θ.forward(micro_batch)
          r(θ) = exp(log_prob_new - old_log_prob)

          # PPO Clipped Loss
          L1 = -A · r(θ)
          L2 = -A · clip(r(θ), 1-ε, 1+ε)
          L_clip = max(L1, L2)

          # Dual-clip (for negative advantages)
          L3 = -A · 3.0
          L_dual = min(L3, L_clip)  when A < 0

          L_pg = L_dual (A<0) or L_clip (A≥0)

          # Entropy bonus
          L_ent = -α_ent · H[π_θ]

          # KL loss
          L_KL = α_KL · KL_low_var(π_θ, π_ref)

          L_total = L_pg + L_ent + L_KL
          L_total.backward()
        clip_grad_norm(1.0)
        optimizer_θ.step()

    ========== 阶段 5: 验证 & 日志 ==========

    14. 每 5 步执行验证 (128 个验证 episode)
    15. 记录 metrics: reward, advantage, loss, grad_norm, ...
```

---

## 14. ALFWorld 示例详解

### 14.1 环境描述

ALFWorld 是一个基于文本的家居任务环境。Agent 需要在虚拟房间中完成任务，如"把苹果放到冰箱里"。

典型交互：

```
[环境观测]
You are in the middle of a room. Looking around you, you see a countertop, 
a fridge, a stove, and a cabinet. Your task is to: put a clean apple in fridge.

[Agent 生成]
go to countertop 1

[环境反馈]
On the countertop 1, you see an apple 1, a knife 1.

[Agent 生成]
take apple 1 from countertop 1

[环境反馈]
You pick up the apple 1 from the countertop 1.
...
```

### 14.2 动作投影

**文件**: `agent_system/environments/env_manager.py:151`

```python
actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
```

将模型自由文本映射到 ALFWorld 可接受的动作集合中。如果生成的文本不在 admissible actions 中，`valids[i] = False`，会受到 `invalid_action_penalty_coef=0.1` 的惩罚。

### 14.3 观测模板

**文件**: `agent_system/environments/env_manager.py:180-212`

环境管理器将原始观测包装为结构化 prompt：

- **初始步**: 使用 `ALFWORLD_TEMPLATE_NO_HIS` 模板，只有当前观测和可用动作
- **后续步**: 使用 `ALFWORLD_TEMPLATE` 模板，包含：
  - 任务描述
  - 当前步数
  - 最近 `history_length=2` 步的历史（观测+动作）
  - 当前观测
  - 可用动作列表

### 14.4 成功率评估

**文件**: `agent_system/environments/env_manager.py:214-242`

```python
def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
    # 找到最后一个活跃步
    for i in reversed(range(len(total_batch_list[batch_idx]))):
        if batch_item['active_masks']:
            won_value = float(info['won'])
            success['success_rate'].append(won_value)
            # 按任务类型分别统计
            self._process_gamefile(gamefile, won_value, success)
```

ALFWorld 的 6 种任务类型：
1. `pick_and_place` — 拿起放下
2. `pick_two_obj_and_place` — 拿起两个物体放下
3. `look_at_obj_in_light` — 在灯下查看物体
4. `pick_heat_then_place_in_recep` — 加热后放置
5. `pick_cool_then_place_in_recep` — 冷却后放置
6. `pick_clean_then_place_in_recep` — 清洁后放置

### 14.5 训练数据示例

一个 episode（假设 5 步交互）产生 5 个训练样本：

| Step | Prompt (观测) | Response (动作) | Episode Reward | Active | Is Valid |
|------|--------------|-----------------|----------------|--------|----------|
| 1 | "You are in the room..." | "go to countertop 1" | 1.0 | True | True |
| 2 | "On countertop, you see..." | "take apple 1" | 1.0 | True | True |
| 3 | "You pick up apple..." | "go to sinkbasin" | 1.0 | True | True |
| 4 | "You are at sinkbasin..." | "clean apple 1" | 1.0 | True | True |
| 5 | "You clean apple 1..." | "put apple 1 in fridge 1" | 1.0 | True | True |

注意：
- 每个 step 的 `episode_rewards` 都是相同的 (episode 级别)
- 奖励只放在每个 step 的 response 最后一个 token 位置
- GAE 使用 Critic 的 value 估计来分配 credit

### 14.6 数据大小变化

```
初始 batch: 128 个环境
     │
     ▼  (每个环境平均 ~10 步交互)
有效步数: ~1280 条训练记录
     │
     ▼  (adjust_batch 调整为 size_divisor 的倍数)
调整后: 1280+ 条记录 (可能复制少量样本补齐)
     │
     ▼  (分成 mini-batches, 每个 256/world_size=128 条)
Mini-batches: ~10 个 mini-batches
     │
     ▼  (每个分成 micro-batches, 每个 16 条)
Micro-batches: 每个 mini-batch ~8 个 micro-batches
```

---

## 附录: 关键函数索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 入口函数 | `verl/trainer/main_ppo.py` | 29 |
| 初始化组件 | `verl/trainer/main_ppo.py` | 56-188 |
| 训练主循环 `fit()` | `verl/trainer/ppo/ray_trainer.py` | 1007-1304 |
| 多轮交互入口 | `agent_system/multi_turn_rollout/rollout_loop.py` | 484-539 |
| 单次 rollout 循环 | `agent_system/multi_turn_rollout/rollout_loop.py` | 285-414 |
| 观测预处理 | `agent_system/multi_turn_rollout/rollout_loop.py` | 43-188 |
| 轨迹数据收集 | `agent_system/multi_turn_rollout/rollout_loop.py` | 233-283 |
| Episode 奖励计算 | `agent_system/reward_manager/episode.py` | 29-96 |
| 无效动作惩罚 | `verl/trainer/ppo/ray_trainer.py` | 200-224 |
| KL 惩罚 (reward 中) | `verl/trainer/ppo/ray_trainer.py` | 152-198 |
| 优势计算分发 | `verl/trainer/ppo/ray_trainer.py` | 244-362 |
| GAE 实现 | `verl/trainer/ppo/core_algos.py` | 67-109 |
| GRPO 实现 | `verl/trainer/ppo/core_algos.py` | 113-174 |
| PPO Clipped Loss | `verl/trainer/ppo/core_algos.py` | 431-492 |
| Dual-clip PPO | `verl/trainer/ppo/core_algos.py` | 485-489 |
| Value Loss | `verl/trainer/ppo/core_algos.py` | 580-612 |
| KL 散度 (多种类型) | `verl/trainer/ppo/core_algos.py` | 615-648 |
| Actor log_prob 计算 | `verl/workers/actor/dp_actor.py` | 252-314 |
| Actor 策略更新 | `verl/workers/actor/dp_actor.py` | 316-447 |
| Actor 前向传播 | `verl/workers/actor/dp_actor.py` | 76-232 |
| Critic value 计算 | `verl/workers/critic/dp_critic.py` | 138-177 |
| Critic 更新 | `verl/workers/critic/dp_critic.py` | 179-260 |
| ALFWorld 环境管理 | `agent_system/environments/env_manager.py` | 133-242 |
| 环境工厂函数 | `agent_system/environments/env_manager.py` | 602-698 |
| Batch 大小调整 | `agent_system/multi_turn_rollout/utils.py` | 86-130 |
| 动态采样过滤 | `agent_system/multi_turn_rollout/utils.py` | 133-184 |
| Worker 初始化 | `verl/trainer/ppo/ray_trainer.py` | 828-909 |
| 数据指标计算 | `verl/trainer/ppo/metric_utils.py` | 79-189 |
| Loss 聚合 | `verl/trainer/ppo/core_algos.py` | 395-428 |
| 自适应 KL 控制器 | `verl/trainer/ppo/core_algos.py` | 29-44 |
| 默认配置 | `verl/trainer/config/ppo_trainer.yaml` | 1-317 |
