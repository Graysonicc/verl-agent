# PRADA-lite 方法设计与代码实现说明

本文说明本仓库中新增的 PRADA-lite 实现，重点对应 `prada.md` 中的 no-head PRADA-lite 公式，并标明每个公式模块在代码中的位置。

## 1. 使用方式

在现有 ALFWorld GRPO 脚本基础上，只需要把 advantage estimator 改成：

```bash
algorithm.adv_estimator=prada_lite
```

例如可以运行：

```bash
bash examples/grpo_trainer/run_alfworld.sh vllm algorithm.adv_estimator=prada_lite
```

新增默认超参位于 `verl/trainer/config/ppo_trainer.yaml:255-269`，可以用 Hydra override 调整，例如：

```bash
algorithm.prada_lite.top_k=32 \
algorithm.prada_lite.bootstrap_rounds=32 \
algorithm.prada_lite.temporal_band=3 \
algorithm.prada_lite.lambda_u=0.3
```

默认 `algorithm.prada_lite.representation=policy_hidden`，会在 actor 重算 old log prob 时额外取最后层 hidden states 做 pooling。若要退回最低开销表示，可设：

```bash
algorithm.prada_lite.representation=hashed_bow
```

## 2. 设计选择

`prada.md` 中 PRADA-lite 的理想形式使用冻结 prefix encoder \(f(h_{i,t})\)。本实现把主路径改成 **policy hidden-state pooling**：

\[
z_{i,t}=\mathrm{Normalize}\left(
\frac{\sum_{\ell}m_{\ell}h_{\theta}(x_{i,t})_{\ell}}
{\sum_{\ell}m_{\ell}}
\right)
\]

这里 \(h_\theta\) 是当前 actor backbone 的最后层 hidden states，\(m_\ell\) 是 attention mask。它不新增预测头、不新增监督，只复用 rollout 后本来就会执行的 actor log-prob forward。

为了保持鲁棒，本实现仍保留 token hashed bag-of-words fallback：如果没有拿到 `prada_prefix_embeddings`，PRADA-lite 会退回用当前 rollout batch 已经有的 token：

- `prompts`：每个环境 step 的 observation prompt。
- `responses`：模型在该 step 生成的 `<think>/<action>` response。
- `attention_mask`：过滤 padding token。
- `uid`：同一原始 prompt 的 GRPO group id。
- `traj_uid`：每条环境轨迹 id。
- `episode_rewards`：trajectory outcome reward。

fallback 表示为：

\[
z_{i,t} = \mathrm{HashBoW}(\mathrm{tokens}(h_{i,t}, a_{i,t}))
\]

这个 fallback 更便宜，但现在只是备用路径和消融项，不是默认主方法。

## 3. 新增与修改文件

新增文件：

- `prada/__init__.py:1-5`：暴露 PRADA 模块。
- `prada/core_prada.py:1-381`：PRADA-lite 核心实现。
- `PRADA_LITE_IMPLEMENTATION.md`：本文档。

修改文件：

- `verl/trainer/ppo/ray_trainer.py:64`：导入 `core_prada`。
- `verl/trainer/ppo/ray_trainer.py:85-98`：新增 `AdvantageEstimator.PRADA_LITE = "prada_lite"`。
- `verl/trainer/ppo/ray_trainer.py:362-389`：新增 PRADA-lite advantage 分支。
- `verl/trainer/ppo/ray_trainer.py:475-486`：PRADA-lite 不使用 critic，与 GRPO/GiGPO 一样走 actor-only RL。
- `verl/trainer/ppo/ray_trainer.py:1267-1283`：把 `algorithm.prada_lite.*` 超参传入 advantage 计算，并把 PRADA metrics 写入训练日志。
- `verl/trainer/config/ppo_trainer.yaml:255-269`：新增 PRADA-lite 默认超参。
- `verl/workers/actor/dp_actor.py:76-81`：新增 hidden-state mean pooling。
- `verl/workers/actor/dp_actor.py:83-95`：运行时检查模型 forward 是否支持 `output_hidden_states` 或 `**kwargs`，不支持则自动 fallback。
- `verl/workers/actor/dp_actor.py:147-149` 和 `:236-241`：需要 PRADA embedding 时请求 `output_hidden_states=True`。
- `verl/workers/actor/dp_actor.py:198-213` 和 `:258-259`：从最后层 hidden states 生成 `prefix_embeddings`。
- `verl/workers/fsdp_workers.py:694-701`：把 `prada_prefix_embeddings` 随 old log prob 一起返回。

## 4. 公式到代码的逐项对应

### 4.1 超参数

公式中的 \(K,B,\Delta,\tau,\alpha_0,\beta_0,\lambda_u,\gamma_\rho,\lambda_\rho,\epsilon\) 对应 `PradaLiteConfig`：

- 代码：`prada/core_prada.py:13-28`
- 默认配置：`verl/trainer/config/ppo_trainer.yaml:255-269`

含义：

- `top_k` 对应 top-\(K\) 近邻数。
- `bootstrap_rounds` 对应 bootstrap 轮数 \(B\)。
- `temporal_band` 对应时间窗口 \(\Delta\)。
- `tau` 对应近邻权重温度 \(\tau\)。
- `alpha0/beta0` 对应 Beta prior。
- `lambda_u` 对应不确定性下降项系数 \(\lambda_u\)。
- `trace_gamma/trace_lambda` 对应 \(\gamma_\rho,\lambda_\rho\)。
- `epsilon` 对应数值稳定项。
- `representation` 控制 prefix 表示。默认 `policy_hidden`；`hashed_bow` 会跳过 hidden-state 保存并使用 token HashBoW。
- `hash_dim` 是 fallback hashed token feature 维度。
- `min_success_reward` 可把非二值 reward 映射成 success label。

### 4.2 Prefix 表示 \(z_{i,t}=f(h_{i,t})\)

`prada.md` 公式：

\[
z_{i,t}=f(h_{i,t})
\]

主路径代码实现：

- `ray_trainer.py:1175-1180`：当 `adv_estimator=prada_lite` 且 `representation=policy_hidden` 时，给 `compute_log_prob` 设置 `return_prada_prefix_embeddings=True`。
- `dp_actor.py:76-81`：对最后层 hidden states 做 attention-mask mean pooling 和 L2 normalize。
- `dp_actor.py:83-95`：用 `inspect.signature` 检查 forward 参数，避免不支持 `output_hidden_states` 的 custom wrapper 报错。
- `dp_actor.py:147-149`、`:236-241`：actor forward 请求 `output_hidden_states=True`。
- `dp_actor.py:198-213`、`:258-259`：兼容 remove-padding 和普通 forward 两条路径，生成 `prefix_embeddings`。
- `fsdp_workers.py:694-701`：把 pooled embedding 写入 tensor 字段 `prada_prefix_embeddings`。
- `ray_trainer.py:371`：PRADA advantage 分支把 `prada_prefix_embeddings` 传入核心函数。
- `core_prada.py:184-187`：核心函数优先使用 dense policy hidden embeddings。

fallback 代码实现：

- `_valid_token_list`：`prada/core_prada.py:38-44`
- `_hashed_bow`：`prada/core_prada.py:47-57`
- `_build_prefix_features`：`prada/core_prada.py:155-169`

实现说明：

1. 默认从 actor hidden states 得到 dense \(z_{i,t}\)。
2. 如果没有 dense embedding，则从 `prompts` 和 `responses` 中取有效 token，使用 `attention_mask` 去掉 padding。
3. prompt 最多保留后 512 个 token，response 最多保留 128 个 token，避免长 prompt 造成 KNN 过慢。
4. 将 token id 映射到 `hash_dim` 桶并做 L2 归一化，得到 sparse vector。

这个模块对应 PRADA-lite 的“冻结表示”部分，但不训练新 encoder。

### 4.3 同 prompt/task 近邻权重 \(\omega\)

`prada.md` 公式：

\[
\omega_{i,t}^{j,u}
=
\exp\left(\frac{\cos(z_{i,t},z_{j,u})}{\tau}\right)
\mathbf 1[|u-t|\le \Delta]
\]

代码实现：

- dense hidden-state cosine：`prada/core_prada.py:184-187` 和 `:205-207`
- sparse fallback cosine：`prada/core_prada.py:60-65`
- step index 计算：`prada/core_prada.py:68-74`
- group 内 temporal band + top-K 检索：`prada/core_prada.py:172-208`

实现说明：

- 只在同一个 `uid` 内找近邻，保持“同一 prompt/task rollout group”约束。
- 用 `traj_uid` 的出现顺序构造每条轨迹的 step index。
- 只保留 \(|u-t|\le\Delta\) 的候选，避免早期 prefix 和晚期 prefix 被错误匹配。
- 对 top-K cosine 相似度应用 \(\exp(\cos/\tau)\) 得到权重。

### 4.4 平滑成功率代理 \(\hat p_{i,t}\)

`prada.md` 公式：

\[
\hat p_{i,t}
=
\frac{\alpha_0 + \sum_{j,u}\omega_{i,t}^{j,u}R_j}
{\alpha_0+\beta_0+\sum_{j,u}\omega_{i,t}^{j,u}}
\]

代码实现：

- outcome reward 提取：`prada/core_prada.py:77-82`
- success label 映射：`prada/core_prada.py:85-94`
- weighted success rate：`prada/core_prada.py:134-139`
- local proxy 调用：`prada/core_prada.py:333-344`

实现说明：

- 优先使用 rollout 中已有 `episode_rewards` 生成 success label，由 trainer 在 `ray_trainer.py:372` 传入。
- 如果没有 `episode_rewards`，退化为从 `token_level_rewards` 按 `traj_uid` 聚合。
- 对 ALFWorld 这种 reward 为 `10 * won` 的环境，默认 `reward > 0` 视为 success；如果是非二值环境，可设置 `algorithm.prada_lite.min_success_reward`。
- 使用 `alpha0/beta0` 做平滑，避免全 0 或全 1 近邻导致 logit 爆炸。

### 4.5 Bootstrap 方差 \(\hat v_{i,t}\)

`prada.md` 公式：

\[
\hat v_{i,t}
=
\mathrm{Var}_{b=1}^{B}\big(\hat p_{i,t}^{(b)}\big)
\]

代码实现：

- `prada/core_prada.py:116-152`

实现说明：

- 对 top-K 近邻按权重归一化后进行有放回重采样。
- 每次重采样重新计算带 Beta prior 的成功率。
- 方差下限为 `epsilon`，避免投影时权重为 0。
- 当 `bootstrap_rounds <= 1` 或只有 1 个近邻时，退化为基于有效样本数的 Bernoulli variance 近似，见 `prada/core_prada.py:141-146`。

### 4.6 轨迹级 group advantage \(A_i^{grp}\)

`prada.md` 公式：

\[
A_i^{grp}=\frac{R_i-\mu_G}{\sigma_G+\epsilon}
\]

代码实现：

- `prada/core_prada.py:97-113`
- 调用位置：`prada/core_prada.py:333-335`

实现说明：

- 按 `uid` 分组。
- 每条 `traj_uid` 只取一次 terminal reward 参与 group mean/std，避免长轨迹因为 step 更多而重复加权。
- 这里的 reward 来自 `token_level_rewards` 按 `traj_uid` 聚合，因此已有的 invalid action penalty、KL-in-reward 等 reward shaping 会继续影响 \(A_i^{grp}\)。
- 计算出的 \(A_i^{grp}\) 会广播到该 trajectory 的所有 active step。

### 4.7 局部责任证据 \(e_{i,t}^{lite}\)

`prada.md` 公式：

\[
e_{i,t}^{lite}
=
\mathrm{logit}(\hat p_{i,t})-\mathrm{logit}(\hat p_{i,t-1})
+\lambda_u\mathrm{sign}(A_i^{grp})(\hat v_{i,t-1}-\hat v_{i,t})
\]

代码实现：

- previous step 对齐：`prada/core_prada.py:222-228`
- logit 与 evidence 计算：`prada/core_prada.py:346-352`

实现说明：

- 第一项衡量 success proxy 的上升或下降。
- 第二项把“不确定性下降”按 outcome 方向对齐：成功轨迹中不确定性下降是正证据，失败轨迹中不确定性下降可能强化负证据。
- 每条轨迹第一步的 previous value 使用 Beta prior 的默认 logit 和 variance。

### 4.8 Trace smoothing \(\tilde e_{i,t}\)

`prada.md` 公式：

\[
\tilde e_{i,t}
=
\sum_{u=t}^{T_i}(\gamma_\rho\lambda_\rho)^{u-t}e_{i,u}
\]

代码实现：

- `_trace_smooth`：`prada/core_prada.py:211-219`
- 调用位置：`prada/core_prada.py:354-355`

实现说明：

- 按 `traj_uid` 从后往前递推。
- 传播的是 responsibility evidence，而不是环境 reward。
- `trace_decay = trace_gamma * trace_lambda`。

### 4.9 闭式受约束投影 \(\hat A_{i,t}^{lite}\)

`prada.md` 公式：

\[
\hat A_{i,t}^{lite}
=
\tilde e_{i,t}
+
w_{i,t}
\frac{T_iA_i^{grp}-\sum_u\tilde e_{i,u}}
{\sum_u w_{i,u}},
\quad
w_{i,t}=\hat v_{i,t}+\epsilon
\]

代码实现：

- `_closed_form_projection`：`prada/core_prada.py:231-248`
- 调用位置：`prada/core_prada.py:356-357`

实现说明：

- 对每条 `traj_uid` 单独投影。
- `target_sum = T_i * A_i^{grp}` 实现全局一致性约束。
- 修正项按 \(w_{i,t}\) 分配，因此更不确定的 step 承担更多全局纠偏。
- 输出满足：

\[
\frac{1}{T_i}\sum_t \hat A_{i,t}^{lite}=A_i^{grp}
\]

### 4.10 写回 token-level advantages

代码实现：

- `prada/core_prada.py:361-362`

实现说明：

- PRADA-lite 先得到 step-level scalar advantage。
- 由于现有 actor loss 接收 token-level `advantages`，实现用：

\[
\mathrm{advantages}_{i,tok}=\hat A_{i,t}^{lite}\cdot\mathrm{response\_mask}_{i,tok}
\]

这与现有 GRPO/GiGPO 的 token-level mask 写法一致。

### 4.11 PRADA 诊断指标

代码实现：

- `prada/core_prada.py:363-380`
- trainer 写入日志：`verl/trainer/ppo/ray_trainer.py:1282-1283`

新增指标：

- `prada_lite/proxy_success_mean`
- `prada_lite/proxy_success_std`
- `prada_lite/bootstrap_var_mean`
- `prada_lite/effective_neighbors_mean`
- `prada_lite/used_policy_hidden_embeddings`
- `prada_lite/evidence_mean`
- `prada_lite/projected_step_adv_mean`
- `prada_lite/global_consistency_error`

其中 `global_consistency_error` 对应：

\[
\left|A_i^{grp}-\frac1{T_i}\sum_t \hat A_{i,t}\right|
\]

正常情况下应接近 0；如果开启 `algorithm.prada_lite.normalize=True`，因为投影后又做额外归一化，该指标可能不再为 0，所以默认关闭。

## 5. Trainer 接入逻辑

PRADA-lite 与 GRPO/GiGPO 一样不需要 critic。

- enum 注册：`verl/trainer/ppo/ray_trainer.py:85-98`
- no-critic 分支：`verl/trainer/ppo/ray_trainer.py:475-486`
- advantage 分支：`verl/trainer/ppo/ray_trainer.py:362-389`

训练流保持不变：

1. rollout 收集环境轨迹。
2. reward manager 写入 `token_level_scores`。
3. invalid action penalty 可继续生效。
4. `token_level_rewards` 生成后调用 PRADA-lite。
5. actor 使用现有 PPO clipped objective 更新。

这保证了 PRADA-lite 是 advantage operator 的替换，不改 rollout、不改 reward manager、不改 actor loss。

## 6. 鲁棒性处理

实现中加入了以下防护：

- 输入 shape 检查：`prada/core_prada.py:301-304`。
- index 长度检查：`prada/core_prada.py:323-326`。
- `episode_rewards` 类型转换报错信息：`prada/core_prada.py:31-35`。
- 无近邻时使用 Beta prior：`prada/core_prada.py:123-125`。
- 权重非有限或全 0 时退化成均匀权重：`prada/core_prada.py:127-131`。
- success label 对二值 reward、ALFWorld reward、阈值 reward 均有兼容：`prada/core_prada.py:85-94`。
- 方差和分母都用 `epsilon` 防止除零：`prada/core_prada.py:141-152`、`:243-247`。

## 7. 与 `prada.md` 的差异

`prada.md` 中写到 \(z_{i,t}=f(h_{i,t})\) 可以复用当前策略模型或冻结底模 hidden state。当前实现已经把 policy hidden-state pooling 作为默认主路径，因此论文叙事可以写成：

\[
z_{i,t}=\mathrm{Pool}(h_\theta(x_{i,t}))
\]

与 PRADA-full 的区别是：这里只复用 actor backbone 表示，不训练 success posterior head。`hashed_bow` 仅作为无 hidden-state 环境或消融实验的 fallback。

## 8. 验证情况

已做静态语法检查：

```bash
env PYTHONPYCACHEPREFIX=/private/tmp/prada_pycache \
  python3 -m py_compile prada/core_prada.py prada/__init__.py \
    verl/trainer/ppo/ray_trainer.py verl/workers/actor/dp_actor.py \
    verl/workers/fsdp_workers.py
```

当前沙箱的系统 Python 没有安装 `torch`，因此没有运行真实训练或 tensor 单元测试。建议在项目训练环境中先用小 batch 运行：

```bash
bash examples/grpo_trainer/run_alfworld.sh vllm \
  algorithm.adv_estimator=prada_lite \
  trainer.total_epochs=1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=False
```
