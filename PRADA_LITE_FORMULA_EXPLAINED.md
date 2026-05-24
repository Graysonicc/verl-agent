# PRADA-lite 公式逐项解释

本文只解释 PRADA-lite 的数学公式、符号含义和下标含义。目标是回答：每个 turn 的 advantage 是怎么从 trajectory reward 分配出来的。

## 1. 一个具体例子

假设同一个 ALFWorld 任务 prompt 下，我们采样 `env.rollout.n=4` 条轨迹：

```text
G = {τ_1, τ_2, τ_3, τ_4}
```

其中：

- \(G\)：同一个 prompt/task 下的一组 rollout。
- \(\tau_i\)：第 \(i\) 条轨迹。
- \(i\)：轨迹下标，例如 \(i=1\) 表示第一条 rollout。
- \(t\)：turn/step 下标，例如 \(t=3\) 表示这条轨迹的第 3 次环境交互。
- \(T_i\)：第 \(i\) 条轨迹总共有多少个有效 turn。

例如：

```text
τ_1: turn 1, turn 2, turn 3, 成功 R_1 = 10
τ_2: turn 1, turn 2, turn 3, turn 4, 失败 R_2 = 0
τ_3: turn 1, turn 2, 成功 R_3 = 10
τ_4: turn 1, turn 2, turn 3, 失败 R_4 = 0
```

PRADA-lite 的目标是：不要把 \(R_i\) 简单复制给每个 turn，而是给每个 turn 一个不同的局部 advantage：

```text
τ_1: Â_1,1, Â_1,2, Â_1,3
τ_2: Â_2,1, Â_2,2, Â_2,3, Â_2,4
...
```

其中：

- \(\hat A_{i,t}\)：第 \(i\) 条轨迹第 \(t\) 个 turn 的最终 advantage。

## 2. 轨迹级 group advantage

公式：

\[
A_i^{grp}
=
\frac{R_i-\mu_G}{\sigma_G+\epsilon}
\]

其中：

- \(A_i^{grp}\)：第 \(i\) 条轨迹的 group-level advantage。
- \(R_i\)：第 \(i\) 条轨迹的最终 reward。
- \(\mu_G\)：同一 rollout group \(G\) 内所有轨迹 reward 的均值。
- \(\sigma_G\)：同一 rollout group \(G\) 内所有轨迹 reward 的标准差。
- \(\epsilon\)：很小的正数，防止除以 0。
- 上标 `grp` 表示这是 group-normalized trajectory advantage。

例子里 reward 是：

```text
R = [10, 0, 10, 0]
```

所以：

\[
\mu_G = \frac{10+0+10+0}{4}=5
\]

\[
\sigma_G = 5
\]

于是：

\[
A_1^{grp}=\frac{10-5}{5}=1
\]

\[
A_2^{grp}=\frac{0-5}{5}=-1
\]

直觉：

- 成功轨迹比组平均好，所以 \(A_i^{grp}>0\)。
- 失败轨迹比组平均差，所以 \(A_i^{grp}<0\)。
- 但这个值仍然是整条轨迹级别的，还没有分到每个 turn。

## 3. 每个 turn 的表示 \(z_{i,t}\)

公式：

\[
z_{i,t}
=
\mathrm{Pool}(h_\theta(x_{i,t}))
\]

更具体地，本实现默认用 mean pooling：

\[
z_{i,t}
=
\mathrm{Normalize}
\left(
\frac{\sum_{\ell}m_{\ell}h_\theta(x_{i,t})_{\ell}}
{\sum_{\ell}m_{\ell}}
\right)
\]

其中：

- \(z_{i,t}\)：第 \(i\) 条轨迹第 \(t\) 个 turn 的向量表示。
- \(x_{i,t}\)：第 \(i\) 条轨迹第 \(t\) 个 turn 实际送进 actor 的文本序列，通常是当前 observation prompt 加本 turn 的 response。
- \(h_\theta(\cdot)\)：当前 actor 模型的 hidden-state 函数。
- \(\theta\)：actor 模型参数。
- \(h_\theta(x_{i,t})_{\ell}\)：输入序列第 \(\ell\) 个 token 的最后层 hidden state。
- \(\ell\)：token 位置下标。
- \(m_\ell\)：attention mask。有效 token 为 1，padding token 为 0。
- \(\sum_\ell m_\ell h_\theta(x_{i,t})_\ell\)：把有效 token 的 hidden states 加起来。
- \(\sum_\ell m_\ell\)：有效 token 数。
- `Normalize`：L2 归一化，方便后续用 cosine similarity。

直觉：

- \(z_{i,t}\) 是这个 turn 的“语义状态 + 动作”的表示。
- 两个 turn 的 \(z\) 越相似，说明它们处在相似的局部情况，或者采取了相似行为。

注意：

- 第 4 轮的 \(x_{i,4}\) 不一定包含完整第 1、2、3、4 轮历史。
- 它包含多少历史，取决于环境 prompt 构造器，例如 ALFWorld 默认 `env.history_length=2`。

## 4. 相似 turn 的近邻权重 \(\omega\)

公式：

\[
\omega_{i,t}^{j,u}
=
\exp
\left(
\frac{\cos(z_{i,t},z_{j,u})}{\tau}
\right)
\mathbf 1[|u-t|\le \Delta]
\]

其中：

- \(\omega_{i,t}^{j,u}\)：当前 turn \((i,t)\) 对候选 turn \((j,u)\) 的近邻权重。
- \(i\)：当前轨迹下标。
- \(t\)：当前 turn 下标。
- \(j\)：候选轨迹下标。
- \(u\)：候选 turn 下标。
- \(z_{i,t}\)：当前 turn 的向量表示。
- \(z_{j,u}\)：候选 turn 的向量表示。
- \(\cos(z_{i,t},z_{j,u})\)：两个向量的 cosine similarity。
- \(\tau\)：温度系数。越小，越偏向最相似的邻居；越大，权重更平均。
- \(\mathbf 1[\cdot]\)：指示函数。条件成立为 1，不成立为 0。
- \(\Delta\)：temporal band，限制只能比较时间步相近的 turn。

为什么有 \(|u-t|\le\Delta\)？

因为第 1 步和第 10 步就算文本相似，也可能处在完全不同的任务阶段。temporal band 用来避免这种错配。

例子：

假设当前是 \(\tau_1\) 的第 2 个 turn，也就是 \((i,t)=(1,2)\)。如果 \(\Delta=1\)，它只会考虑候选 turn 的 \(u\in\{1,2,3\}\)，不会去匹配第 6 步、第 10 步。

直觉：

- \(\omega\) 越大，说明候选 turn 和当前 turn 越相似。
- 后面估计成功率时，相似 turn 的最终成败会有更大影响。

## 5. top-K 近邻集合 \(\mathcal N_{i,t}\)

公式中常写：

\[
\mathcal N_{i,t}
=
\mathrm{TopK}_{j,u}
\left(
\omega_{i,t}^{j,u}
\right)
\]

其中：

- \(\mathcal N_{i,t}\)：当前 turn \((i,t)\) 的 top-K 近邻集合。
- \(K\)：最多保留多少个相似 turn。
- \((j,u)\in\mathcal N_{i,t}\)：表示第 \(j\) 条轨迹第 \(u\) 个 turn 是当前 turn 的一个近邻。

例子：

如果 `top_k=3`，当前 turn \((1,2)\) 的近邻可能是：

```text
N_1,2 = {(1,2), (3,2), (4,1)}
```

这表示它最像：

- 自己这一步 \((1,2)\)
- 第 3 条轨迹第 2 步 \((3,2)\)
- 第 4 条轨迹第 1 步 \((4,1)\)

## 6. 成功率代理 \(\hat p_{i,t}\)

公式：

\[
\hat p_{i,t}
=
\frac{
\alpha_0+\sum_{(j,u)\in\mathcal N_{i,t}}\omega_{i,t}^{j,u}Y_j
}{
\alpha_0+\beta_0+\sum_{(j,u)\in\mathcal N_{i,t}}\omega_{i,t}^{j,u}
}
\]

其中：

- \(\hat p_{i,t}\)：当前 turn \((i,t)\) 的非参数成功率代理。
- \(\alpha_0\)：Beta prior 中的成功伪计数。
- \(\beta_0\)：Beta prior 中的失败伪计数。
- \(\mathcal N_{i,t}\)：当前 turn 的近邻集合。
- \(\omega_{i,t}^{j,u}\)：近邻权重。
- \(Y_j\)：第 \(j\) 条轨迹是否最终成功。成功为 1，失败为 0。

注意这里用的是 \(Y_j\)，不是 \(Y_{j,u}\)。原因是：

- 每个近邻 turn \((j,u)\) 属于第 \(j\) 条轨迹。
- 这个 turn 本身没有独立标签。
- 它继承所在轨迹的最终成败标签 \(Y_j\)。

例子：

假设当前 turn 的 3 个近邻是：

```text
(1,2): 来自成功轨迹 τ_1, Y_1=1, weight=2.0
(3,2): 来自成功轨迹 τ_3, Y_3=1, weight=1.5
(4,1): 来自失败轨迹 τ_4, Y_4=0, weight=0.5
```

设 \(\alpha_0=1,\beta_0=1\)，则：

\[
\hat p_{1,2}
=
\frac{1+2.0\cdot1+1.5\cdot1+0.5\cdot0}
{1+1+2.0+1.5+0.5}
=
\frac{4.5}{6}
=0.75
\]

直觉：

- 如果一个 turn 更像成功轨迹里的 turn，\(\hat p\) 高。
- 如果更像失败轨迹里的 turn，\(\hat p\) 低。

## 7. Bootstrap 方差 \(\hat v_{i,t}\)

公式：

\[
\hat v_{i,t}
=
\mathrm{Var}_{b=1}^{B}
\left(
\hat p_{i,t}^{(b)}
\right)
\]

其中：

- \(\hat v_{i,t}\)：当前 turn 的成功率代理不确定性。
- \(B\)：bootstrap 重采样次数。
- \(b\)：第 \(b\) 次 bootstrap。
- \(\hat p_{i,t}^{(b)}\)：第 \(b\) 次重采样近邻后得到的成功率估计。
- \(\mathrm{Var}\)：方差。

怎么重采样？

对 \(\mathcal N_{i,t}\) 里的近邻有放回采样，采样 \(B\) 次。每次重新计算一个 \(\hat p\)。如果这些 \(\hat p\) 差异大，说明当前 turn 的近邻证据不稳定。

直觉：

- \(\hat v_{i,t}\) 大：当前 turn 的成功倾向不确定，近邻里成功失败混杂。
- \(\hat v_{i,t}\) 小：近邻证据一致，当前 turn 的成功倾向比较确定。

## 8. 局部责任证据 \(e_{i,t}^{lite}\)

公式：

\[
e_{i,t}^{lite}
=
\underbrace{
\mathrm{logit}(\hat p_{i,t})
-
\mathrm{logit}(\hat p_{i,t-1})
}_{\text{成功倾向变化}}
+
\lambda_u
\underbrace{
\mathrm{sign}(A_i^{grp})
\left(
\hat v_{i,t-1}-\hat v_{i,t}
\right)
}_{\text{结果方向一致的不确定性下降}}
\]

其中：

- \(e_{i,t}^{lite}\)：第 \(i\) 条轨迹第 \(t\) 个 turn 的局部责任证据。
- \(\hat p_{i,t}\)：当前 turn 的成功率代理。
- \(\hat p_{i,t-1}\)：同一条轨迹上一个 turn 的成功率代理。
- \(\mathrm{logit}(p)=\log\frac{p}{1-p}\)。
- \(\hat v_{i,t}\)：当前 turn 的 bootstrap 方差。
- \(\hat v_{i,t-1}\)：上一个 turn 的 bootstrap 方差。
- \(\lambda_u\)：不确定性下降项的权重。
- \(\mathrm{sign}(A_i^{grp})\)：轨迹整体结果方向。成功轨迹通常为正，失败轨迹通常为负。

第一项：

\[
\mathrm{logit}(\hat p_{i,t})-\mathrm{logit}(\hat p_{i,t-1})
\]

表示这个 turn 之后，局部状态是否更像成功。

- 如果 \(\hat p\) 上升，这一项为正。
- 如果 \(\hat p\) 下降，这一项为负。

第二项：

\[
\lambda_u\mathrm{sign}(A_i^{grp})(\hat v_{i,t-1}-\hat v_{i,t})
\]

表示不确定性是否朝着最终结果方向被解决。

- 成功轨迹中，不确定性下降是好事。
- 失败轨迹中，不确定性下降可能表示模型更确定地走向失败，因此是负方向证据。

例子：

假设成功轨迹 \(\tau_1\) 中：

```text
p_hat_1,1 = 0.50
p_hat_1,2 = 0.75
v_hat_1,1 = 0.08
v_hat_1,2 = 0.03
A_1^grp > 0
lambda_u = 0.2
```

那么：

\[
\mathrm{logit}(0.75)-\mathrm{logit}(0.50)
=1.0986-0
=1.0986
\]

不确定性项：

\[
0.2\cdot 1\cdot(0.08-0.03)=0.01
\]

所以：

\[
e_{1,2}^{lite}=1.1086
\]

说明第 2 个 turn 是明显正责任。

## 9. Trace smoothing \(\tilde e_{i,t}\)

公式：

\[
\tilde e_{i,t}
=
\sum_{u=t}^{T_i}
(\gamma_\rho\lambda_\rho)^{u-t}
e_{i,u}^{lite}
\]

其中：

- \(\tilde e_{i,t}\)：平滑后的责任证据。
- \(e_{i,u}^{lite}\)：第 \(u\) 个 turn 的原始局部责任证据。
- \(u\)：从当前 turn \(t\) 到轨迹末尾 \(T_i\) 的未来 turn 下标。
- \(T_i\)：第 \(i\) 条轨迹的总 turn 数。
- \(\gamma_\rho\)：责任 trace 的 discount。
- \(\lambda_\rho\)：责任 trace 的 lambda。
- \((\gamma_\rho\lambda_\rho)^{u-t}\)：未来证据传播回当前 turn 的衰减系数。

直觉：

如果第 5 步才看到明显成功信号，第 3、4 步可能也有责任。trace smoothing 会把后面的证据向前传播一点。

例子：

假设一条轨迹有 3 个 turn：

```text
e_1 = 0.2
e_2 = 0.5
e_3 = 1.0
gamma_rho * lambda_rho = 0.8
```

那么：

\[
\tilde e_3 = 1.0
\]

\[
\tilde e_2 = 0.5 + 0.8\cdot1.0 = 1.3
\]

\[
\tilde e_1 = 0.2 + 0.8\cdot0.5 + 0.8^2\cdot1.0 = 1.24
\]

## 10. 不确定性权重 \(w_{i,t}\)

公式：

\[
w_{i,t}=\hat v_{i,t}+\epsilon
\]

其中：

- \(w_{i,t}\)：投影阶段的 correction weight。
- \(\hat v_{i,t}\)：bootstrap 方差。
- \(\epsilon\)：防止为 0。

直觉：

- \(w\) 越大，说明这个 turn 越不确定。
- PRADA 的全局纠偏会更多落到高不确定 turn 上。

这不是说高不确定 turn 一定 advantage 大，而是说当局部证据总和与轨迹级结果不一致时，高不确定 turn 更适合被调整。

## 11. 闭式投影：最终 turn advantage

公式：

\[
\hat A_{i,t}^{lite}
=
\tilde e_{i,t}
+
w_{i,t}
\frac{
T_iA_i^{grp}
-
\sum_{u=1}^{T_i}\tilde e_{i,u}
}{
\sum_{u=1}^{T_i}w_{i,u}
}
\]

其中：

- \(\hat A_{i,t}^{lite}\)：最终给第 \(i\) 条轨迹第 \(t\) 个 turn 的 advantage。
- \(\tilde e_{i,t}\)：平滑后的局部责任证据。
- \(w_{i,t}\)：不确定性权重。
- \(T_i\)：第 \(i\) 条轨迹的 turn 数。
- \(A_i^{grp}\)：第 \(i\) 条轨迹的 group advantage。
- \(\sum_{u=1}^{T_i}\tilde e_{i,u}\)：这条轨迹所有 turn 的局部证据总和。
- \(\sum_{u=1}^{T_i}w_{i,u}\)：这条轨迹所有 turn 的不确定性权重总和。

为什么要投影？

因为局部证据 \(\tilde e\) 只是一个 proxy。它可能和最终 trajectory outcome 不一致。PRADA 强制要求：

\[
\frac{1}{T_i}
\sum_{t=1}^{T_i}
\hat A_{i,t}^{lite}
=
A_i^{grp}
\]

也就是说，每条轨迹所有 turn 的平均 advantage 必须等于这条轨迹的全局 group advantage。

公式里的 correction 项：

\[
\frac{
T_iA_i^{grp}
-
\sum_{u=1}^{T_i}\tilde e_{i,u}
}{
\sum_{u=1}^{T_i}w_{i,u}
}
\]

表示“还差多少全局结果需要补回来”。

然后乘 \(w_{i,t}\)，表示：

- 不确定性大的 turn 多承担纠偏。
- 不确定性小、证据更可靠的 turn 少被改动。

## 12. 投影公式的具体数字例子

假设某条成功轨迹 \(\tau_1\) 有 3 个 turn：

```text
A_1^grp = 1.0
T_1 = 3
```

PRADA 要求最终三个 turn advantage 的平均值为 1：

\[
\frac{\hat A_{1,1}+\hat A_{1,2}+\hat A_{1,3}}{3}=1
\]

也就是总和必须为：

\[
T_1A_1^{grp}=3
\]

假设 trace 后的证据是：

```text
ē = [0.2, 1.4, 0.8]
```

总和：

\[
\sum_u \tilde e_{1,u}=2.4
\]

还差：

\[
3-2.4=0.6
\]

假设不确定性权重：

```text
w = [0.1, 0.3, 0.2]
```

总权重：

\[
\sum_u w_{1,u}=0.6
\]

每单位权重 correction：

\[
\frac{0.6}{0.6}=1
\]

所以最终：

\[
\hat A_{1,1}=0.2+0.1\cdot1=0.3
\]

\[
\hat A_{1,2}=1.4+0.3\cdot1=1.7
\]

\[
\hat A_{1,3}=0.8+0.2\cdot1=1.0
\]

检查平均值：

\[
\frac{0.3+1.7+1.0}{3}=1.0
\]

直觉：

- 第 2 个 turn 局部证据强，所以 advantage 高。
- 第 1 个 turn 局部证据弱，所以 advantage 低。
- 整条轨迹又是成功轨迹，所以三者平均仍严格等于 \(A_1^{grp}=1\)。

## 13. 从 turn advantage 到 token advantage

公式：

\[
A_{i,t,k}^{token}
=
\hat A_{i,t}^{lite}
\cdot
M_{i,t,k}^{resp}
\]

其中：

- \(A_{i,t,k}^{token}\)：第 \(i\) 条轨迹第 \(t\) 个 turn 中第 \(k\) 个 response token 的 advantage。
- \(k\)：response 内部 token 下标。
- \(\hat A_{i,t}^{lite}\)：这个 turn 的 step-level advantage。
- \(M_{i,t,k}^{resp}\)：response mask。有效 response token 为 1，padding token 为 0。

因此：

- 同一个 turn 内，所有有效 response token 的 advantage 相同。
- 不同 turn 的 advantage 可以不同。
- padding token 的 advantage 是 0。

例子：

如果第 2 个 turn 的 response 有 5 个有效 token，\(\hat A_{1,2}=1.7\)，那么：

```text
token advantages = [1.7, 1.7, 1.7, 1.7, 1.7, 0, 0, ...]
```

## 14. 整体计算顺序

PRADA-lite 的公式顺序可以压缩成：

```text
1. 得到每条轨迹终局 reward R_i
2. 在同组内算 trajectory-level A_i^grp
3. 对每个 turn 算 hidden-state 表示 z_i,t
4. 用 z_i,t 找同组、相近时间步的 top-K 近邻
5. 用近邻轨迹的最终成败算 p_hat_i,t
6. bootstrap 得到不确定性 v_hat_i,t
7. 用 p_hat 的变化和 v_hat 的变化算局部证据 e_i,t
8. 对 e_i,t 做 backward trace smoothing 得到 ē_i,t
9. 用闭式投影得到最终 turn advantage Â_i,t
10. 把 Â_i,t 复制给该 turn 内所有有效 response token
```

## 15. 为什么这个方法能处理“成功轨迹里的坏 turn”

普通 GRPO 会给成功轨迹里的所有 turn 类似正向信号。PRADA-lite 不一样。

假设一条成功轨迹里第 2 个 turn 是绕路：

```text
turn 1: 找到正确房间
turn 2: 做了一个无关 examine
turn 3: 拿起正确物体
turn 4: 放到正确位置，成功
```

如果第 2 个 turn 的 \(z_{i,2}\) 更像失败轨迹里的绕路 step，那么：

- \(\hat p_{i,2}\) 可能下降。
- \(e_{i,2}^{lite}\) 可能为负。
- 投影后 \(\hat A_{i,2}\) 也可能低，甚至为负。

但因为整条轨迹成功，PRADA 仍保证所有 turn 的平均 advantage 等于正的 \(A_i^{grp}\)。这就是“全局一致、局部可变号”的含义。

