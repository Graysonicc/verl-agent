# Turn-Level PPO: 缺陷分析与改进建议

---

## 1. Reward 设计的问题

### 1.1 当前现状

目前的 reward 流程存在一个根本性问题：**所有 turn 收到的是同一个 episode-level reward，而非 per-step reward。**

具体链路：
1. `envs.step()` 返回 per-step reward `r_k`，存入 `non_tensor_batch['rewards']`
2. `episode_rewards` 在每一步累加：`episode_rewards += rewards`
3. `gather_rollout_data()` 将**同一个** `episode_rewards`（累加总和）赋给 episode 内的**每个** turn
4. `EpisodeRewardManager` 将这个相同的 `episode_rewards` 放到每个 turn 最后一个有效 token 上

结果：**Turn-Level GAE 中每个 turn 的 reward $r_k$ 实际上是 episode 总奖励 $R$，而非该 turn 的真实即时奖励。**

这意味着 GAE 的 TD-error 公式：

$$\delta_k = r_k + \gamma V(s_{k+1}) - V(s_k)$$

退化为：

$$\delta_k = R + \gamma V(s_{k+1}) - V(s_k)$$

所有 turn 的 $r_k$ 相同，GAE 的信用分配完全依赖 $V(s_{k+1}) - V(s_k)$ 的差值，丧失了 per-step reward 提供的直接监督信号。

### 1.2 为什么问题严重

| 场景 | 影响 |
|------|------|
| **sparse reward 环境** (ALFWorld: 成功=1, 失败=0) | 所有 turn 收到相同的 0 或 1，GAE 退化为纯 value-difference 驱动，与 token-level PPO 的退化方式相同——turn-level 的优势被抵消 |
| **dense reward 环境** (Sokoban: 每步有奖惩) | per-step reward 本应是 turn-level GAE 最大的优势来源，但被 episode-level 赋值抹杀 |
| **长 episode** (50 步) | $R$ 被重复 50 次，梯度信号中 reward 项的方差为零（相对于 episode 间而言是常量），GAE 的信用分配能力大幅下降 |

### 1.3 改进方案

#### 方案 A：使用 per-step reward（推荐）

`non_tensor_batch['rewards']` 已经存储了 `envs.step()` 返回的 per-step reward，但 `EpisodeRewardManager` 忽略了它。

改进：新建 `TurnLevelRewardManager`，当 `turn_level.enable=True` 时：
- 每个 turn 的 reward 使用 `non_tensor_batch['rewards']`（即 `envs.step()` 的原始返回值）
- 仅放在该 turn 最后一个有效 token 位置
- 不再使用 episode-level 的统一赋值

```python
# 改进后的 reward 赋值
for i in range(len(data)):
    step_reward = data_item.non_tensor_batch['rewards']  # per-step reward
    reward_tensor[i, valid_response_length - 1] = step_reward
```

**注意**：对于 sparse reward 环境 (ALFWorld)，每步的 `rewards` 实际上大多为 0，只有最终步为 1（成功）或始终为 0（失败）。因此即使用了 per-step reward，GAE 的信号依然稀疏。这不是 bug，而是 sparse reward 的固有特性——对此需要搭配方案 B。

#### 方案 B：Reward Shaping / 辅助奖励

对 sparse reward 环境，仅靠最终一步的 0/1 信号做 turn-level GAE 效果有限。可以引入辅助 reward：

| 辅助 reward | 说明 | 适用环境 |
|-------------|------|---------|
| **Invalid action penalty** | 已实现 (`invalid_action_penalty_coef`)，但作为 token_level_scores 的修正，未参与 turn-level reward | ALFWorld |
| **Progress reward** | 基于子任务进度给予中间奖励（如 ALFWorld 中完成一个子步骤 +0.1） | 有子任务结构的环境 |
| **Exploration bonus** | 对访问新状态给予小额正奖励 | 状态空间大的环境 |
| **Potential-based shaping** | $F(s, s') = \gamma \Phi(s') - \Phi(s)$，不改变最优策略 | 通用，需要设计势函数 |

#### 方案 C：混合 Reward 策略

将 episode-level reward 和 per-step reward 加权组合：

$$r_k^{\text{mixed}} = \alpha \cdot r_k^{\text{step}} + (1 - \alpha) \cdot \frac{R}{K}$$

其中 $R$ 是 episode 总奖励，$K$ 是 turn 数，$\alpha$ 是混合系数。这样在 per-step reward 为 0 时仍有来自 episode reward 的均匀分配信号。

---

## 2. 分段粒度的问题

### 2.1 当前 turn 的定义

当前 turn 的定义是**一次完整的模型生成 → 环境反馈循环**，即 `vanilla_multi_turn_loop` 中 `for _step in range(max_steps)` 的一次迭代。

这个定义存在以下粒度问题：

### 2.2 粒度过粗的情况

| 场景 | 问题 |
|------|------|
| **一个 turn 中包含多次工具调用** | 某些 agent 框架中，一次模型生成可能包含 `<think>...</think><tool_call>...</tool_call>` 多个语义段，但被视为一个 turn |
| **思考 + 行动混合** | 模型输出 "Let me think... I should go to countertop 1. go to countertop 1"，思考部分和行动部分语义不同，但共享同一个 advantage |
| **长 response** | 某些 turn 的 response 很长（如生成代码块），内部可能包含多个独立决策点 |

### 2.3 粒度过细的情况

| 场景 | 问题 |
|------|------|
| **连续的信息收集 turn** | agent 连续执行 "look", "examine table", "open drawer" 等探索动作，这些属于同一个决策阶段（探索），单独给每个 turn 一个 advantage 可能引入噪声 |
| **环境反馈无信息量** | 某些 turn 的环境反馈是 "Nothing happens"（无效动作），这些 turn 的 value 估计本身缺乏意义，但仍然参与 GAE 计算 |

### 2.4 改进方案

#### 方案 A：Sub-turn 分段

将一个 turn 内部按**语义标记**进一步分段：

```
Turn 1 (模型生成): "<think>I need to find a mug</think> go to countertop 1"
  → Segment 1 (think):  tokens [t1, ..., t8]   → V_think, A_think
  → Segment 2 (action): tokens [t9, ..., t13]  → V_action, A_action
```

实现方式：
- 在 response 中检测特殊标记（`<think>`, `<tool_call>`, action delimiter 等）
- 对每个 segment 独立计算 value 和 advantage
- 需要修改 `_apply_turn_level_values` 支持多个 segment per turn

**优点**：更精细的信用分配，尤其对 think-then-act 模式  
**缺点**：依赖特定的输出格式，通用性差；实现复杂度高

#### 方案 B：Phase-level 分段

将多个连续的 turn 按**任务阶段**合并：

```
Phase 1 (探索): [turn 1: "look", turn 2: "examine table", turn 3: "open drawer"]
Phase 2 (执行): [turn 4: "take mug", turn 5: "go to sink"]
Phase 3 (完成): [turn 6: "clean mug", turn 7: "put mug on shelf"]
```

实现方式：
- 基于 reward 变化或环境状态变化检测 phase 边界
- 同一 phase 内的 turn 共享 advantage
- GAE 在 phase 级别计算

**优点**：减少探索阶段的噪声  
**缺点**：phase 边界的自动检测困难；对于简单环境可能不需要

#### 方案 C：自适应粒度（推荐）

保持 turn 为基本单元，但引入**注意力权重机制**来决定每个 turn 的 advantage 贡献：

$$\hat{A}_k^{\text{weighted}} = w_k \cdot \hat{A}_k^{\text{turn}} + (1 - w_k) \cdot \hat{A}_k^{\text{token}}$$

其中 $w_k$ 根据该 turn 的特征自适应调整：
- $w_k \to 1$：当 turn 是短的原子动作（如 "go to table 1"）
- $w_k \to 0$：当 turn 包含复杂思考链（长 response、多语义段）

$w_k$ 可以通过以下方式计算：
- 基于 response 长度：$w_k = \sigma(-\beta \cdot (n_k - \bar{n}))$，短 turn 权重高
- 基于 token-level value 方差：turn 内部 value 方差大说明 token-level 信号有用，应降低 $w_k$

---

## 3. Critic 估值的问题

### 3.1 Last-token 提取的局限

当前使用最后一个有效 token 的 Critic 输出作为 turn value。问题：

| 问题 | 分析 |
|------|------|
| **位置偏差** | 最后一个 token 的 hidden state 可能过度关注局部上下文（如 action 的最后一个词），而非整个 turn 的全局语义 |
| **训练信号稀疏** | Critic 仅在最后一个 token 位置接收梯度，中间 token 的 Critic 输出不参与 loss，可能导致 Critic 学习效率低 |
| **长 turn 的信息丢失** | 对于 100+ token 的 response，最后一个 token 的 representation 能力有限 |

### 3.2 改进方案

#### 方案 A：Mean Pooling（配置已支持但未实现）

用 turn 内所有有效 token 的 Critic 输出均值作为 turn value：

$$V(s_k) = \frac{1}{n_k} \sum_{j=1}^{n_k} V_\phi(s_k, a_k^{\leq j})$$

**优点**：梯度信号回传到所有 token，训练更稳定  
**缺点**：混合了不同位置的信息，可能引入噪声

#### 方案 B：Attention Pooling

引入一个轻量的 attention 层，对 turn 内所有 token 的 value 做加权平均：

$$V(s_k) = \sum_{j=1}^{n_k} \alpha_j \cdot V_\phi(s_k, a_k^{\leq j}), \quad \alpha_j = \text{softmax}(W \cdot h_j)$$

**优点**：自动学习哪些 token 的估值更重要  
**缺点**：引入额外参数，增加模型复杂度

#### 方案 C：双头 Critic

保留 token-level Critic head，同时新增一个 turn-level Critic head（类似 reward model 的做法）：

- Token-level head：在所有 token 上计算 loss（辅助任务，保持 representation 质量）
- Turn-level head：仅在 turn 边界计算 loss（主任务，用于 GAE）

$$L_{\text{critic}} = L_{\text{turn}} + \beta \cdot L_{\text{token}}$$

**优点**：token-level 辅助任务提供丰富的训练信号，turn-level head 专注于 turn value  
**缺点**：增加计算量和参数量

---

## 4. IS Ratio 的问题

### 4.1 几何平均模式的梯度传播

当前实现使用 GSPO-style 的梯度技巧：

```python
turn_log_ratio_with_grad = log_prob - log_prob.detach() + turn_log_ratio.detach()
```

这使得每个 token 收到的梯度与其自身的 log_prob 变化成比例，但 **clipping 是基于 turn-level ratio 的**。这意味着：

- 如果 turn-level ratio 被 clip，turn 内**所有** token 的梯度同时被截断
- 即使某个 token 的 individual ratio 很温和，如果其他 token 导致 turn-level ratio 超阈值，它也被连带 clip
- 反之，某个 token 的 ratio 极端，但被 turn 平均稀释，可能逃过 clipping

### 4.2 改进方案

#### 方案 A：双层 Clipping

同时在 turn-level 和 token-level 做 clip：

$$\rho_j^{\text{final}} = \text{clip}\left(\rho_k^{\text{geo}}, 1-\epsilon_{\text{turn}}\right) \cdot \text{clip}\left(\frac{\rho_j}{\rho_k^{\text{geo}}}, 1-\epsilon_{\text{token}}\right)$$

内层 clip 限制单 token 偏离 turn 平均的程度，外层 clip 限制整个 turn 的策略变化幅度。

#### 方案 B：自适应 Clip 范围

根据 turn 长度动态调整 clip 范围：

$$\epsilon_k = \epsilon_0 \cdot \sqrt{n_k}$$

长 turn 用更宽的 clip 范围（因为几何平均本身已经做了归一化），短 turn 用更紧的范围。

---

## 5. Turn-Level GAE 的结构性问题

### 5.1 同一 episode 内的 turn 数差异

不同 episode 的 turn 数可能差异极大（1 步完成 vs 50 步超时）。这导致：

- **短 episode** (K=1-2)：GAE 几乎无法传播信用信号，退化为 MC return
- **长 episode** (K=50)：GAE 在 $\gamma\lambda < 1$ 时信号快速衰减，早期 turn 的 advantage 接近 0

### 5.2 改进方案

- **对短 episode**：fallback 到 MC return（已在设计文档中提到），具体实现为：当 $K \leq 2$ 时，直接使用 $\hat{A}_k = R_k - V(s_k)$
- **对长 episode**：考虑使用 $\gamma = 1.0, \lambda = 0.95$ 的配置以保留更多远期信号，或者使用分段 GAE（在 phase boundary 处重启 GAE 计算）

---

## 6. 改进优先级建议

| 优先级 | 改进项 | 原因 | 预期收益 |
|--------|--------|------|---------|
| **P0** | 使用 per-step reward 替代 episode-level reward | 当前实现的根本缺陷，直接影响 GAE 正确性 | 高 |
| **P1** | 实现 mean pooling Critic | 低实现成本，缓解 last-token 偏差和梯度稀疏问题 | 中 |
| **P1** | 短 episode fallback | 简单实现，避免 K=1 时的退化 | 中 |
| **P2** | 混合 reward 策略 | 对 sparse reward 环境提供基础信号 | 中 |
| **P2** | 自适应 turn/token 混合 advantage | 灵活处理不同复杂度的 turn | 中 |
| **P3** | Sub-turn 分段 | 依赖输出格式，通用性受限 | 低-中 |
| **P3** | 双层 clipping | 实现复杂，需要仔细调参 | 低 |
| **P3** | 双头 Critic | 增加模型复杂度，需要更多实验验证 | 低 |

---

## 7. 总结

Turn-Level PPO 的核心方向是正确的——将 PPO 的计算粒度对齐到 agent-environment 交互的自然单元。但当前实现存在一个 **P0 级缺陷**（reward 赋值方式使得 turn-level GAE 丧失了 per-step 信用分配能力）和若干 **P1/P2 级可优化项**（Critic 提取方式、分段粒度、IS clipping 策略）。

最关键的改进路径：
1. 先修 reward（P0），让 per-step reward 正确流入 GAE
2. 实现 mean pooling Critic（P1），改善 value estimation
3. 在消融实验中逐步验证各项改进的独立贡献
