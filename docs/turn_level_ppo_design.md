# Turn-Level PPO for Agentic Reinforcement Learning

> 一种将 PPO 的核心计算粒度从 token 级别提升到 turn 级别的算法设计，专为 Agent-Environment 多轮交互场景设计。

---

## 1. 动机 (Motivation)

### 1.1 问题分析

在标准的 Agentic RL (如 verl-agent 中的 PPO) 中，尽管 Agent 与环境的交互以 **turn（轮次）** 为单位——每个 turn 是一次完整的观测-动作循环——但 PPO 的三个核心计算却在 **token 级别** 进行：

| 组件 | 当前粒度 | 问题 |
|------|---------|------|
| **Critic (价值估计)** | 每个 token 一个 $V(s_t)$ | Agent 动作内部的 token 并不对应独立的状态转移，逐 token 估值语义模糊 |
| **Advantage (优势估计)** | 每个 token 一个 $\hat{A}_t$ | 一个 turn 内部各 token 的优势值不同，但实际上同一个 turn 对应同一个「动作决策质量」 |
| **Importance Sampling (IS)** | 每个 token 一个 $r_t(\theta)$ | token 级 IS 比率的乘积在长 response 上会指数级偏离 1，导致高方差 |

**核心矛盾：** Agentic RL 中环境反馈的最小语义单元是 turn（一个完整的动作执行），而非 token。Token 级别的 PPO 在以下方面产生不匹配：

1. **信用分配模糊**：一个 turn 生成 "go to countertop 1"，其中 "go"、"to"、"countertop"、"1" 各有不同的 advantage，但实际上只有整个动作序列作为整体才具有语义意义。
2. **Critic 估值噪声**：Critic 被迫对动作内部的中间 token 进行价值估计，而这些中间状态并无环境交互，估值缺乏有效的监督信号。
3. **IS 比率发散**：假设一个 turn 包含 $n$ 个 token，token 级 IS 比率的乘积 $\prod_{j=1}^n r_j$ 容易偏离 1，在 $n$ 较大时导致梯度方差急剧增大，PPO clip 频繁触发但效果有限。

### 1.2 核心思想

将 PPO 的三个核心计算从 token 级别对齐到 turn 级别，使算法粒度与环境交互的语义单元一致：

```
标准 PPO (Token-Level):
  Step 1 (Turn 1):  [t1, t2, t3, t4]  →  V(t1), V(t2), V(t3), V(t4)
  Step 2 (Turn 2):  [t5, t6, t7]       →  V(t5), V(t6), V(t7)

Turn-Level PPO (本方案):
  Step 1 (Turn 1):  [t1, t2, t3, t4]  →  V(Turn 1) → 共享给所有 token
  Step 2 (Turn 2):  [t5, t6, t7]       →  V(Turn 2) → 共享给所有 token
```

---

## 2. 算法设计

### 2.1 符号定义

| 符号 | 含义 |
|------|------|
| $K$ | 一个 episode 中的总 turn 数 |
| $k \in \{1, ..., K\}$ | 第 $k$ 个 turn |
| $n_k$ | 第 $k$ 个 turn 中的 response token 数 |
| $s_k$ | 第 $k$ 个 turn 的状态（即该 turn 的环境观测） |
| $a_k = (a_k^1, a_k^2, ..., a_k^{n_k})$ | 第 $k$ 个 turn 中生成的 token 序列（即完整动作） |
| $r_k$ | 第 $k$ 个 turn 执行后获得的环境奖励 |
| $V(s_k)$ | Turn 级别的状态价值估计 |
| $\hat{A}_k$ | Turn 级别的优势估计 |
| $\gamma$ | 折扣因子 |
| $\lambda$ | GAE 的 lambda 参数 |

### 2.2 Turn 边界检测

在 verl-agent 的 `vanilla_multi_turn_loop` 中，每一步 `for _step in range(max_steps)` 产生一个独立的 `(prompt, response)` pair，存入 `total_batch_list[bs]`。因此：

- **每个 step 恰好对应一个 turn**，turn 边界天然由环境交互步定义
- 训练数据中 `traj_uid` 标识同一个 episode 的所有 turn
- 同一 episode 的 turn 按时间顺序排列

Turn 边界信息可以通过以下方式获取：

1. 同一 `traj_uid` 的数据属于同一 episode
2. 同一 episode 内按 `step` 顺序排列
3. 每条数据记录即为一个完整的 turn

### 2.3 Turn-Level Critic

#### 设计思路

原始 Critic 对每个 token 位置输出一个标量价值估计 $V(s_t) \in \mathbb{R}$，形状为 `(batch_size, response_length)`。Turn-Level Critic 改为对每个 turn 输出一个标量价值估计 $V(s_k) \in \mathbb{R}$，然后广播到该 turn 内所有 token。

#### 价值提取方式

取 Critic 模型在 **该 turn response 最后一个有效 token** 位置的输出作为 turn 级价值估计。这与语言模型中常用的 sequence-level representation 提取方式一致（如 reward model 的做法）。

$$V(s_k) = V_\phi^{[\text{last\_valid\_token}]}(s_k, a_k)$$

其中 $V_\phi^{[\text{last\_valid\_token}]}$ 表示取 Critic 输出在 response 最后一个有效 token 位置的值。

#### 广播规则

同一 turn 内所有 token 共享同一个 value：

$$v_t = V(s_k) \quad \forall t \in \text{Turn}_k$$

#### Forward 伪代码

```python
def compute_turn_values(self, data: DataProto) -> torch.Tensor:
    """
    计算 turn 级别的 value，并广播到所有 token。
    
    Returns:
        values: (batch_size, response_length) — 同一 turn 内所有 token 共享同一个 value
    """
    # 原始逐 token 的 Critic forward
    token_values = self._forward_micro_batch(micro_batch)  # (bs, resp_len)
    
    # 取每条数据最后一个有效 token 的 value 作为 turn value
    response_mask = attention_mask[:, -response_length:]
    valid_lengths = response_mask.sum(dim=-1).long()  # (bs,)
    
    turn_values = torch.zeros(batch_size, device=token_values.device)
    for i in range(batch_size):
        last_idx = valid_lengths[i] - 1
        turn_values[i] = token_values[i, last_idx]
    
    # 广播: 将 turn value 复制到该 turn 的所有有效 token
    values = turn_values.unsqueeze(-1).expand(-1, response_length) * response_mask
    
    return values
```

#### Turn-Level Critic Update

更新时，仅在每个 turn 的最后一个有效 token 位置计算 value loss，而非所有 token：

```python
def compute_turn_value_loss(vpreds, returns, values, response_mask, valid_lengths, cliprange_value):
    """
    仅在每个 turn 的最后一个有效 token 位置计算 value loss。
    """
    batch_size = vpreds.shape[0]
    
    # 构造 turn-level mask: 仅最后一个有效 token 为 1
    turn_mask = torch.zeros_like(response_mask)
    for i in range(batch_size):
        last_idx = valid_lengths[i] - 1
        turn_mask[i, last_idx] = 1.0
    
    vpredclipped = clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    
    # 仅在 turn 边界位置计算 loss
    vf_loss = (clipped_vf_losses * turn_mask).sum() / turn_mask.sum()
    
    return vf_loss
```

### 2.4 Turn-Level Advantage Estimation

#### Turn-Level GAE

在 turn 级别应用 GAE。将同一 episode 的所有 turn 按时间顺序排列后：

$$\delta_k = r_k + \gamma \cdot V(s_{k+1}) - V(s_k) \quad (k < K)$$
$$\delta_K = r_K - V(s_K) \quad (\text{最后一个 turn})$$

$$\hat{A}_k^{\text{Turn-GAE}} = \sum_{l=0}^{K-k} (\gamma\lambda)^l \delta_{k+l}$$

$$\text{Returns}_k = \hat{A}_k + V(s_k)$$

#### 广播规则

Turn 级优势估计广播到该 turn 内所有 token：

$$\hat{A}_t = \hat{A}_k \quad \forall t \in \text{Turn}_k$$

#### 白化处理

对广播后的 advantage 做白化（zero-mean, unit-variance），以稳定训练。白化在所有有效 token 上计算统计量。

#### 实现伪代码

```python
def compute_turn_level_gae(
    token_level_rewards: torch.Tensor,   # (total_steps_in_batch, resp_len)
    turn_values: torch.Tensor,           # (total_steps_in_batch,), 每个 turn 一个 value
    response_mask: torch.Tensor,         # (total_steps_in_batch, resp_len)
    traj_uids: np.ndarray,              # (total_steps_in_batch,), episode 标识
    gamma: float = 1.0,
    lam: float = 1.0,
):
    """
    在 turn 级别计算 GAE，然后广播到 token 级别。
    """
    # 1. 计算每个 turn 的总奖励 (token 奖励之和)
    turn_rewards = (token_level_rewards * response_mask).sum(dim=-1)  # (total_steps,)
    
    # 2. 按 episode 分组，每组内按时间顺序排列
    advantages = torch.zeros_like(turn_rewards)
    returns = torch.zeros_like(turn_rewards)
    
    unique_trajs = np.unique(traj_uids)
    for traj_id in unique_trajs:
        mask = (traj_uids == traj_id)
        indices = np.where(mask)[0]  # 该 episode 的所有 turn 索引
        
        K = len(indices)
        lastgaelam = 0.0
        
        for t_rev in range(K - 1, -1, -1):  # 从最后一个 turn 反向遍历
            idx = indices[t_rev]
            if t_rev < K - 1:
                next_idx = indices[t_rev + 1]
                next_value = turn_values[next_idx]
            else:
                next_value = 0.0  # 终止状态
            
            delta = turn_rewards[idx] + gamma * next_value - turn_values[idx]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages[idx] = lastgaelam
            returns[idx] = lastgaelam + turn_values[idx]
    
    # 3. 广播 advantage 到 token 级别
    token_advantages = advantages.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    token_returns = returns.unsqueeze(-1).expand_as(token_level_rewards) * response_mask
    
    # 4. 白化
    token_advantages = masked_whiten(token_advantages, response_mask)
    
    return token_advantages, token_returns
```

### 2.5 Turn-Level Importance Sampling (IS)

这是本方案最灵活的部分，提供三种模式来控制 IS 比率的计算粒度。

#### 基本定义

对于 turn $k$ 中的每个 token $j$，token 级 IS 比率为：

$$r_j = \frac{\pi_\theta(a_k^j \mid s_k, a_k^{<j})}{\pi_{\theta_{old}}(a_k^j \mid s_k, a_k^{<j})} = \exp\left(\log\pi_\theta(a_k^j) - \log\pi_{\theta_{old}}(a_k^j)\right)$$

---

#### 模式一：几何平均 (Geometric Mean)

**直觉**：对 turn 内所有 token 的 IS 比率取几何平均，等价于对 log-ratio 取算术平均后求指数。每个 token 共享相同的 IS 比率，消除了序列长度对 IS 大小的影响。

$$\rho_k^{\text{geo}} = \left(\prod_{j=1}^{n_k} r_j\right)^{1/n_k} = \exp\left(\frac{1}{n_k} \sum_{j=1}^{n_k} \left[\log\pi_\theta(a_k^j) - \log\pi_{\theta_{old}}(a_k^j)\right]\right)$$

**广播**：Turn 内所有 token 使用相同的 $\rho_k^{\text{geo}}$ 作为 IS 比率。

**PPO 目标**：
$$L_k^{\text{CLIP-geo}} = -\hat{A}_k \cdot \min\left(\rho_k^{\text{geo}},\ \text{clip}\left(\rho_k^{\text{geo}},\ 1-\epsilon,\ 1+\epsilon\right)\right)$$

**性质**：
- $\rho_k^{\text{geo}} \approx 1$ 时策略变化小，与 $n_k$ 无关
- 对长短不同的 turn 有统一的尺度
- 等价于 GSPO 中的 sequence-level importance ratio 归一化思想
- 方差较低，适合 turn 长度差异大的场景

---

#### 模式二：乘积 (Product / Sequence-Level Ratio)

**直觉**：直接使用所有 token IS 比率的乘积，这等价于整个 turn 的序列级 IS 比率。

$$\rho_k^{\text{prod}} = \prod_{j=1}^{n_k} r_j = \exp\left(\sum_{j=1}^{n_k} \left[\log\pi_\theta(a_k^j) - \log\pi_{\theta_{old}}(a_k^j)\right]\right)$$

**广播**：Turn 内所有 token 使用相同的 $\rho_k^{\text{prod}}$。

**PPO 目标**：
$$L_k^{\text{CLIP-prod}} = -\hat{A}_k \cdot \min\left(\rho_k^{\text{prod}},\ \text{clip}\left(\rho_k^{\text{prod}},\ 1-\epsilon,\ 1+\epsilon\right)\right)$$

**性质**：
- 理论上最精确的 turn 级 IS，精确衡量策略在整个动作序列上的概率比
- 当 $n_k$ 较大时，$\rho_k^{\text{prod}}$ 可能偏离 1 很远（指数级），导致 clip 频繁触发
- 方差较高，但信号更强
- 可能需要更保守的 clip 范围来稳定训练

---

#### 模式三：逐 Token (Per-Token / Baseline)

**直觉**：保持原始的 token 级 IS 比率不变，仅在 Critic 和 Advantage 上使用 turn 级别。这是一个**混合模式**，可以作为消融实验的基线。

$$\rho_j = r_j = \exp\left(\log\pi_\theta(a_k^j) - \log\pi_{\theta_{old}}(a_k^j)\right)$$

**PPO 目标**（对每个 token 独立计算，但 advantage 已经是 turn 级别广播的）：
$$L_j^{\text{CLIP-token}} = -\hat{A}_k \cdot \min\left(\rho_j,\ \text{clip}\left(\rho_j,\ 1-\epsilon,\ 1+\epsilon\right)\right) \quad \forall j \in \text{Turn}_k$$

**性质**：
- 与原始 PPO 的 IS 机制相同
- 不同 token 的 IS 比率不同，但共享同一个 advantage
- 方差中等，兼容性最好

---

#### 三种模式对比

| 属性 | 模式一 (几何平均) | 模式二 (乘积) | 模式三 (逐 Token) |
|------|-----------------|--------------|------------------|
| IS 粒度 | Turn 级 | Turn 级 | Token 级 |
| 广播 | 是 (同一 turn 共享) | 是 (同一 turn 共享) | 否 (每个 token 独立) |
| 长度敏感性 | 低 (归一化) | 高 (指数增长) | 中 |
| 方差 | 低 | 高 | 中 |
| 理论精确性 | 近似 | 精确 | Token 级精确 |
| 适用场景 | turn 长度差异大 | turn 长度较短且稳定 | 消融基线 |

#### 实现伪代码

```python
def compute_turn_level_is_ratio(
    log_prob: torch.Tensor,         # (bs, resp_len), 当前策略 log prob
    old_log_prob: torch.Tensor,     # (bs, resp_len), 旧策略 log prob
    response_mask: torch.Tensor,    # (bs, resp_len)
    is_mode: str = "geometric",     # "geometric" | "product" | "token"
) -> torch.Tensor:
    """
    计算 IS 比率。返回形状 (bs, resp_len)。
    """
    # token 级 log ratio
    token_log_ratio = log_prob - old_log_prob  # (bs, resp_len)
    
    if is_mode == "token":
        # 模式三: 直接返回 token 级 ratio
        return torch.exp(token_log_ratio)
    
    # 计算 turn 级 log ratio
    valid_lengths = response_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (bs, 1)
    sum_log_ratio = (token_log_ratio * response_mask).sum(dim=-1, keepdim=True)  # (bs, 1)
    
    if is_mode == "geometric":
        # 模式一: 几何平均 → 对 log ratio 取平均
        turn_log_ratio = sum_log_ratio / valid_lengths  # (bs, 1)
    elif is_mode == "product":
        # 模式二: 乘积 → 直接用 log ratio 的和
        turn_log_ratio = sum_log_ratio  # (bs, 1)
    else:
        raise ValueError(f"Unknown IS mode: {is_mode}")
    
    # 广播到所有 token
    turn_ratio = torch.exp(turn_log_ratio).expand_as(log_prob)  # (bs, resp_len)
    
    return turn_ratio
```

### 2.6 Turn-Level PPO Loss 完整公式

综合以上三个组件，Turn-Level PPO 的 Actor Loss 定义如下：

$$L^{\text{Turn-PPO}}(\theta) = L^{\text{policy}} + \alpha_{\text{ent}} \cdot L^{\text{entropy}} + \alpha_{\text{KL}} \cdot L^{\text{KL}}$$

其中策略损失部分（以模式一为例）：

$$L^{\text{policy}} = -\frac{1}{\sum_k n_k} \sum_{k=1}^{K} \sum_{j=1}^{n_k} \min\left(\rho_k^{\text{geo}} \cdot \hat{A}_k,\ \text{clip}(\rho_k^{\text{geo}}, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_k\right)$$

注意 $\hat{A}_k$ 和 $\rho_k^{\text{geo}}$ 对 turn $k$ 内所有 token 是常量，因此：

$$L^{\text{policy}} = -\frac{1}{\sum_k n_k} \sum_{k=1}^{K} n_k \cdot \min\left(\rho_k^{\text{geo}} \cdot \hat{A}_k,\ \text{clip}(\rho_k^{\text{geo}}, 1-\epsilon, 1+\epsilon) \cdot \hat{A}_k\right)$$

这自然引入了 **按 turn 长度加权** 的效果——更长的 turn 对 loss 贡献更大。

---

## 3. 与标准 PPO 的对比

```
标准 Token-Level PPO:
┌─────────────────────────────────────────────────────────────┐
│ Turn 1: tokens [t1, t2, t3, t4]                             │
│   V:   [V(t1), V(t2), V(t3), V(t4)]  ← 4 个不同的 value    │
│   A:   [A(t1), A(t2), A(t3), A(t4)]  ← 4 个不同的 advantage │
│   IS:  [r(t1), r(t2), r(t3), r(t4)]  ← 4 个不同的 ratio     │
│                                                              │
│ Turn 2: tokens [t5, t6, t7]                                  │
│   V:   [V(t5), V(t6), V(t7)]         ← 3 个不同的 value     │
│   A:   [A(t5), A(t6), A(t7)]         ← 3 个不同的 advantage  │
│   IS:  [r(t5), r(t6), r(t7)]         ← 3 个不同的 ratio      │
└─────────────────────────────────────────────────────────────┘

Turn-Level PPO (本方案):
┌─────────────────────────────────────────────────────────────┐
│ Turn 1: tokens [t1, t2, t3, t4]                             │
│   V:   [V(s1), V(s1), V(s1), V(s1)]  ← 1 个共享 value      │
│   A:   [A_1,   A_1,   A_1,   A_1  ]  ← 1 个共享 advantage  │
│   IS:  [ρ_1,   ρ_1,   ρ_1,   ρ_1  ]  ← 1 个共享 ratio      │
│                                         (模式一/二)          │
│                                                              │
│ Turn 2: tokens [t5, t6, t7]                                  │
│   V:   [V(s2), V(s2), V(s2)]         ← 1 个共享 value       │
│   A:   [A_2,   A_2,   A_2  ]         ← 1 个共享 advantage   │
│   IS:  [ρ_2,   ρ_2,   ρ_2  ]         ← 1 个共享 ratio       │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 预期优势与潜在风险

### 4.1 预期优势

1. **更准确的信用分配**：Turn 级别的 GAE 直接在环境交互步之间传播信用信号，避免了 token 内部的虚假信用分配。
2. **更低的估计方差**：
   - Critic 仅需在 turn 边界给出一个估值，减少了不必要的中间 token 估值噪声
   - 几何平均 IS (模式一) 天然具备长度归一化，避免长 turn 中 IS 比率发散
3. **计算效率**：Critic 更新时仅在 turn 边界计算 loss，减少了有效计算量。
4. **语义一致性**：PPO 的每个组件都在与环境交互对齐的粒度上操作。

### 4.2 潜在风险与应对

| 风险 | 分析 | 应对策略 |
|------|------|---------|
| **粒度过粗** | Turn 级别可能丢失 token 内部的有用梯度信号 | 模式三 (token IS) 保留细粒度 IS，仅在 Critic/Adv 层面提升粒度 |
| **乘积 IS 发散 (模式二)** | 长 turn 中 IS 乘积可能极端大或小 | 可以对 $\log\rho_k^{\text{prod}}$ 加 clamp 限制 (如 [-10, 10]) |
| **Critic 学习困难** | 仅用 turn 最后一个 token 的输出做监督，可能导致梯度信号稀疏 | 可选实现：用 turn 内所有 token 的 value 均值作为 turn value (mean pooling)，而非仅取最后一个 |
| **episode 内 turn 数过少** | 若 episode 只有 1-2 个 turn，GAE 无法有效传播 | 结合 episode 奖励直接作为 return 的 fallback |

### 4.3 消融实验建议

| 实验 | 变量 | 目的 |
|------|------|------|
| Baseline | Token-Level Critic + Token-Level Adv + Token-Level IS | 原始 PPO |
| Exp 1 | **Turn-Level Critic** + Token-Level Adv + Token-Level IS | 隔离 Critic 改进效果 |
| Exp 2 | Turn-Level Critic + **Turn-Level Adv** + Token-Level IS | 叠加 Adv 改进 |
| Exp 3a | Turn-Level Critic + Turn-Level Adv + **几何平均 IS** | 完整方案 (模式一) |
| Exp 3b | Turn-Level Critic + Turn-Level Adv + **乘积 IS** | 完整方案 (模式二) |
| Exp 3c | Turn-Level Critic + Turn-Level Adv + **逐 Token IS** | 完整方案 (模式三) |

---

## 5. 配置参数设计

建议在 `ppo_trainer.yaml` 中新增以下配置项：

```yaml
algorithm:
  turn_level:
    enable: True                  # 是否启用 Turn-Level PPO
    critic_mode: "last_token"     # Critic value 提取方式: "last_token" | "mean_pool"
    is_mode: "geometric"          # IS 模式: "geometric" | "product" | "token"
    is_clamp: 10.0                # 对 turn-level log IS ratio 的 clamp 范围 (仅模式二)
    adv_broadcast: True           # 是否将 turn adv 广播到所有 token
    critic_loss_mode: "turn_only" # Critic loss 计算: "turn_only" (仅 turn 边界) | "broadcast" (广播后所有 token)
```

---

## 6. 需要修改的核心代码

| 修改内容 | 文件 | 原始行号 | 改动说明 |
|---------|------|---------|---------|
| Turn-Level Critic Forward | `verl/workers/critic/dp_critic.py` | 138-177 | 新增 `compute_turn_values` 方法 |
| Turn-Level Critic Update | `verl/workers/critic/dp_critic.py` | 179-260 | 新增 turn-level value loss |
| Turn-Level GAE | `verl/trainer/ppo/core_algos.py` | 67-109 | 新增 `compute_turn_level_gae` 函数 |
| Turn-Level IS Ratio | `verl/workers/actor/dp_actor.py` | 316-447 | 在 `update_policy` 中根据 is_mode 计算 IS |
| Advantage 分发 | `verl/trainer/ppo/ray_trainer.py` | 244-362 | 在 `compute_advantage` 中新增 turn-level GAE 分支 |
| Trainer 集成 | `verl/trainer/ppo/ray_trainer.py` | 1007-1304 | 在 `fit()` 中条件调用 turn-level 组件 |
| Turn 元数据传递 | `agent_system/multi_turn_rollout/rollout_loop.py` | 233-283 | 在 `gather_rollout_data` 中记录 turn 序号 |
| 配置项 | `verl/trainer/config/ppo_trainer.yaml` | 234-256 | 新增 `algorithm.turn_level` 配置块 |

---

## 7. 与相关工作的联系

| 工作 | 关系 |
|------|------|
| **GSPO** (Sequence-Level PPO) | 模式一 (几何平均 IS) 与 GSPO 的 sequence-level importance ratio 思想一致，但 GSPO 没有 turn-level GAE |
| **PPO** (Schulman et al., 2017) | 本方案在 token 级别退化为标准 PPO（当 turn 只有 1 个 token 时） |
| **GRPO** (DeepSeek) | GRPO 用 group-level 的 outcome reward 差异代替 GAE，无 Critic。本方案保留 Critic 但提升到 turn 级别 |
| **GiGPO** (本项目) | GiGPO 在 step 级别做 group advantage，与本方案的 turn-level advantage 互补 |
| **Option-Critic** (Bacon et al., 2017) | 类似的层次化 RL 思想，但 Option-Critic 关注选项的发现，本方案关注已有 turn 结构的利用 |

---

## 8. 总结

Turn-Level PPO 的核心是**将 PPO 的三大计算对齐到 Agent-Environment 交互的自然粒度**：

1. **Critic**：一个 turn 一个 value → 减少中间 token 的估值噪声
2. **Advantage**：一个 turn 一个 advantage → 与环境反馈单元一致
3. **IS Ratio**：三种模式灵活选择 → 平衡方差和精确性

这种对齐使得 PPO 在 Agentic RL 场景中更加语义合理，有望带来更稳定的训练和更好的样本效率。
