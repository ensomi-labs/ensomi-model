# 原生生成的 LN 组织与同列压力：问题、机制和改进方向

当前系统尚未达到稳定可玩的约 4★ 生成质量。主要问题包括：LN 的持续与释放缺少可跟随的组织，局部质地被长时间铺满，Stream 条件下出现持续单列攻击，以及整体难度标量接近请求却掩盖局部压力异常。

这些问题不能统一归因于模型太小，也不能用禁止短 LN、强制均匀分列或定期插入休息来解决。诊断需要区分两个条件：**提案分布能否产生合适的完整续写；玩家响应与 continuation 选择能否接受其中合适的续写。** 本文记录已确认的机制、完成的对照和仍待验证的架构／训练假设，没有宣布新的可玩模型。

系统目标是完整音频到实际可玩的 4K、2–6★ 谱面，星级按 osu!mania difficulty algorithm 定义；本报告重点观察约 4★。实验中的全图重算使用仓库的 `compute_mania_star_rating_20241007`，metadata 分层与局部 strain proxy 另行标明，不能混称同一个难度量。训练和推理均可使用完整音频。接口和语义以 [V3 formulation](../formulation/README.md) 为准：提交后的历史不可修改，LN 可以跨发布边界保持开放，控制按各自声明的范围生效。

H 指必须出现至少一个新 head 的时间；head 包括 TAP 和 LN press。R 指纯释放事件的时间；带 head 的行也可以释放其他列。R1 负责完整同时行，包括 head 数、列、TAP/LN 类型和释放哪些 LN。H 不负责和弦大小或分指。

## 原始问题判断及其证据状态

以下判断来自目标提出者对实际生成和谱面创作的要求。它们不是从某个现成指标反推出来的需求。证据栏区分已测机制、同类复现和仍待检验的解释。

| 原始判断／要求 | 对应证据 | 目前能得出的结论 |
| --- | --- | --- |
| 可玩性是最终目标，NLL 和星级只是 proxy | 极端持续单列攻击仍可得到约 4.14 的 scoped proxy；继续训练也出现 NLL 任务与 native 质量分离 | 单一 likelihood／星级不能作晋级条件；不等于正确的联合 likelihood 没有统计意义 |
| 约 4★ 的面条图可以覆盖很高，但 LN 通常有可跟随的 stream／协调组织；生成的细碎感不对 | 2,506 张源谱、108 张高覆盖参考，以及本报告的 LN 关系和 source-H 对照 | 高覆盖和短时长本身不能定义坏；时间关系与持续策略需要分别诊断 |
| Stream 不应退化为极端 long jack，压力应能按实际组织在多指间转移 | 同 H 的 34 heads 中有 31 个落在一列，其他列未被占用；较早端点能分散承担 | 至少存在 R1 偏好／学习退化，不是列资源被用尽；正常带反向声部的 ranked jack 仍须保留 |
| 玩家状态应由已提交历史形成，包含 entry/release、coordination、jack 累积和恢复 | 局部 HH 成本对大于约 107ms 的连续攻击可一直为零；现有 planner 只有攻击 excess 通道 | 当前响应覆盖不完整；增加一个名为 state 的向量并不自动获得这些语义 |
| 缺少多尺度呼吸、variation 和与音乐的呼应；音频进入编排的作用可能有限 | 已有 memory Hysteric 的 16 秒窗口四列攻击为 `[93,97,96,104]`，相关 excess episode 持续 47.662 秒；共享加法条件的主行头存在下述代数限制 | 问题既可能是分配，也可能是所有列都承压；音频“被输入”不证明其能调节所有相对编排偏好。该长窗口尚未做完整 Lens 判断 |
| Attention／更大容量可能必要，但须根据整体职责来决定 | 4.584M→7.617M 的匹配 memory 实验完成后仍未晋级；换 source H 却能改善部分 LN 关系 | 不能仅凭参数少判断根因；目标、信息交互、状态分布和容量应分别验证，允许在证据支持时 scaling |
| 首先要学到广泛真实分布，其次要能采到并接受合适续写 | outcome recipe 只有 47 个固定前缀场景；当前选择器未覆盖 LN／协调响应 | 提案覆盖与选择能力是两个独立可失败的条件，须分别量化 |
| 应由 continuation/frontier 根据后果拒绝坏提案，而非写形态禁令 | 短 LN 在真实高覆盖谱中普遍存在；多数被检查的短尾有继续持有的合法替代 | 不新增时长／图形硬过滤；需要比较具体候选在具体历史下的后果 |
| 控制可独立缺省并在中途限定范围生效；不同范围不能混合抵消 | Stream-only 条件曝光很少；全曲 LN request 被 override 打断后存在评估单位陷阱 | 缺失模式、范围语义和状态连续性属于训练／评估契约，不是 UI 细节 |

最初试玩的 long-jack 文件未保留，后续数字来自同模型谱系、相同控制类型的复现，不声称重建了那次具体输出。上述用户判断、ranked 源谱事实、程序测量和模型侧 Lens 阅读具有不同证据来源；不把它们合并成新的人类标注。

## 对象、空间与概率：不能混为一谈的四层

设完整音频为 $A$，时刻 $t$ 前已公告的 scoped controls 为 $c_{\le t}$。一张谱是有序完整行序列

$$
\mathcal B=((t_i,a_i))_{i=1}^{N},\qquad
a_i\in\{0,\mathrm{tap},\mathrm{press},\mathrm{release}\}^{4},\quad t_{i+1}>t_i.
$$

这里 $0$ 是某列在该行无动作，整行不能全空。已提交历史 $h_t$ 还包括截至 $t$ 的无行决定。精确状态 $x(h_t)$ 由 replay 确定，包括 LN 占用、起点和最近动作时钟；它不是学习得到的难度状态。

需要分别定义：

1. $\mathcal L(h_t,e)$：在 $(t,e]$ 上满足 V3 语法与物理执行的合法未来。私有／发布 horizon 可以保留开放 LN；真正音频 EOF 则必须满足终止闭合规则。统计 scope 另用 $[a,b)$，不混用端点约定。
2. $\mathcal S_\kappa(h_t,e)\subseteq\mathcal L(h_t,e)$：研究 recovery profile $\kappa$ 和实现约束留下的支持。合法空间不应被某次实验的支持反向定义。
3. $q_\theta(Y\mid A,c,h_t,e)$：提案模型在该支持上的概率。概率高表示模型偏好，不直接等于玩家能承受。
4. $\mathcal C_0(h_t,t;Y,e)$：formulation 要表达的 canonical 玩家响应。其值域记为 $\mathcal R$，不预设为一个星级标量。音乐适切性另外依赖 $A$；玩家响应、音乐组织和控制符合度共同参与评价。

因此有三种本质不同的失败：好未来被排除在 $\mathcal S_\kappa$ 外；好未来仍有支持但 $q_\theta$ 几乎不给概率；好未来已经采到，但响应／选择器偏好更差的未来。单看最终输出或一个总分不能识别是哪一种。

完整音频训练／推理对称，表示 $A$ 在两者中都可用；它不意味着历史分布对称。训练常见 $h_t\sim d_{\mathrm{data}}$，部署实际到达 $h_t\sim d_{\pi_\theta}^{\mathrm{BOS}}$。同时，R1 在源谱训练中看到真实 H，在部署中看到生成 H。source-H 干预检查的是这个条件分布差异的一部分，不是在测试 Mel 是否包含足够信息。

### 生成因子与支持条件化

令 $T_H$ 是 H 时间序列，$e_j=(t_j,\zeta_j)$ 是下一个材料化事件及其 H/R 角色，$a_j$ 为完整行。略去确定性的音频编码后，当前模型可写为下式。它针对固定控制日程和当前单向 H 实现；动态请求按公告更新可用信息，不提前读取尚未公告的控制。引入 committed-response feedback 的变体需要重新声明条件依赖。

$$
q_\theta(T_H,\{e_j,a_j\}_j\mid A,c)
=q^H_\theta(T_H\mid A,c)
\prod_j q^{\mathrm{wait}}_\theta(e_j\mid\mathcal I_j,T_H;W_\kappa)
q^1_\theta(a_j\mid\mathcal I_j,e_j,\operatorname{preview}(T_H)).
$$

$\mathcal I_j$ 含当前因子允许读取的该轨迹前缀；私有分支使用自己的已生成历史，不把它写入全局已提交状态。$W_\kappa$ 是由状态与 H 推导的释放窗口。$q^{\mathrm{wait}}$ 包含 R 发生前的 survival，以及直到下一 H 都不发生 R 的概率；不是只给已有 release 行打分。R1 的选择改变下一步 LN 状态和窗口，所以“网络参数分开”不等于“生成过程互不依赖”。H 不直接读 R1 embedding，也不消除这种下游耦合。

若只截取普通 horizon 的轨迹，末尾未发生事件的 survival 项也必须保留。该观察边界不等于真实 EOF，不能用强制 LN 闭合替代右删失。

若原始逐毫秒 release hazard 为 $p_k$，且必须在可行窗 $[a,b]$ 内释放，则第一事件分布是

$$
q_R(t\mid a\le T_R\le b)
=\frac{p_t\prod_{k=a}^{t-1}(1-p_k)}{1-\prod_{k=a}^{b}(1-p_k)}.
$$

因此即使很少命中最后的 $b$，窗口条件化仍可能影响全部早期概率。诊断必须同时记录原始 hazard、条件化窗口和 R1 的释放选择；只数 forced-deadline 事件会漏掉作用。本报告还区分 H 行释放，才得以排除窗口作为所列短尾的普遍充分解释。

### 正确的行概率分解不等于正确的响应语义

令 $m(a)$ 为完整行内部的 head/LN/release 数，$\ell_\theta$ 为布局 logit，$g_\theta$ 为 `frontier2` 分数。其核心组合形式是

$$
q^1_\theta(a\mid u)\propto
p_\theta(m(a)\mid u)
\frac{\exp\ell_\theta(a,u)}{\sum_{b\in\mathcal S_\kappa(u):m(b)=m(a)}\exp\ell_\theta(b,u)}
\exp g_\theta(a,u).
$$

$u$ 汇集该行的合法条件。把 $g$ 放在组内归一化之后是必要的：同组共享代价若放在分子／分母内会抵消。当前实现保留了这个正确结构。

但仅用最终行 likelihood 训练时，$g$ 的“玩家代价”含义没有被识别。一般的 proposal-energy 组合 $\pi\propto q_b\exp g$ 满足

$$
q'_b\propto q_b\exp f,\qquad g'=g-f
\quad\Longrightarrow\quad \pi'=\pi.
$$

这说明在可表达这些重分配的函数范围内，同一个输出分布可以对应不同的内部能量解释。网络被命名为 frontier、或 score 确实改变了概率，都不能代替独立响应目标与验证。

部署还有 neural law 之外的变换。记 recovery preference 作用后的行为分布为 $\bar q$，$\gamma(a)=(n_{\rm head},n_{\rm release})$，$n_{\rm LN}(a)$ 为新 LN 数，则当前 LN feedback 相当于

$$
\pi^1(a)=\bar q(\gamma(a))\,
\frac{\bar q(a\mid\gamma(a))\exp(b_t n_{\rm LN}(a))}
{\mathbb E_{z\sim\bar q(\cdot\mid\gamma(a))}\exp(b_t n_{\rm LN}(z))}.
$$

它保留当前行的 head/release-count 边缘分布，却改变后续占用、release window、行数和未来分布。不能从这一瞬时不变量推出全轨迹 head 数或 workload 不变。基础训练使用的 neural law 与后续 `replay_row_scores` 重构的 deployed law 也应区分；memory384 的源谱行损失使用后者，包含真实前缀重建出的反馈。

## 1. 模块实际做了什么

| 组件 | 当前实际作用 | 尚未由它保证的性质 |
| --- | --- | --- |
| 音频编码与 H | 从完整 Mel、H 历史和控制产生未来 head 时间，满足四列短期容量条件 | 音乐上的持续节奏组织、呼吸和长程负担 |
| R | 从音频、H/R 时间历史、LN 占用／年龄和控制产生纯释放时间；scheduler 提供可行时间窗 | 释放与当前 LN 声部、后续编排的音乐／协调后果相符 |
| R1 | 读取直接音频、完整行历史、精确状态、H preview 和控制，给完整行分布 | 整段 native 续写维持真实 ranked 的组织与可玩性 |
| `RowConsequence(frontier2)` | 为候选行增加随行似然训练的能量项 | 对实际未来续写的独立、已校准玩家响应 |
| `CommittedPlayState` | 保存攻击、释放、持有、近期行和多尺度事实，随真实时间推进 | 这些事实已经等价于协调难度或人类疲劳 |
| 可选 `ResponsePlanner` | 展开若干条私有未来，按持续攻击 excess 选择，提交其中一段前缀 | LN entry/release、协调、音乐组织和所有控制均被评价 |

实现入口：[H/R/R1](../../src/ensomi_model/research/controlled_audio_continuation/model.py)、[native session](../../src/ensomi_model/research/planned_audio_continuation/session.py)、[候选行后果](../../src/ensomi_model/research/bounded_typed_continuation/consequence.py)、[玩家状态](../../src/ensomi_model/research/player_response/state.py)、[continuation planner](../../src/ensomi_model/research/controlled_audio_continuation/frontier.py)。

### Frontier 的功能差距

V3 的 frontier 是同一已提交历史对不同合法未来的响应函数：

$$
\mathcal F_{h_t}(Y,e)=\mathcal C_0(h_t,t;Y,e).
$$

它必须区分具体未来动作序列 $Y$ 和真实时间终点 $e$。空白时间也有意义；持有中的空白不能当作休息。

当前 `frontier2` 的显式特征主要是候选动作后的占用、攻击／释放间隔、LN 年龄，以及推进到前两个 H 的被动时钟。[native 特征构造](../../src/ensomi_model/research/planned_audio_continuation/features.py) 中的 `next_r=now+1` 是可能的下一事件机会，不是预测的释放，更不是已有恢复规则下承诺会发生的释放。该模块没有实际 continuation 输入，也没有独立的响应目标。

其 learned context 可以包含更长历史、音频和控制，因而不能断言它理论上无法识别长 jack。已确认的缺口是：**训练只要求提高观察行的概率，没有要求其分数正确比较未来的累计压力与协调后果。** “frontier2 已存在”不足以证明坏续写会被拒绝。

R/R1 还有一个决策顺序：R 选中纯释放时间后，R1 必须在该事件实现至少一次释放。R1 可选释放哪列，却不能把该事件整体撤回。因此只改行评分无法纠正所有不恰当的 R 时间；私有完整续写的比较应能撤销其中的时间提案。带 H 行的释放是另一条路径，不能把所有 LN 尾归咎于 R。

## 2. 证据和测量定义

固定权重端点分别称为 actor128 与 memory384。前者属于已报告 long-jack 回归的谱系，后者是未晋级的联合音频／history-memory 实验。其初始化、训练及恢复 profile 不同，**不是隔离的容量对照**。因果比较只在各端点内部进行。

| 项目 | actor128 | memory384 |
| --- | --- | --- |
| HH / RH / HR，ms | 60 / 50 / 50 | 60 / 50 / 40 |
| 原生 H 与 source-H 对照 | 各 4 首歌 × 2 seeds | 各 4 首歌 × 2 seeds |

四首歌是 Max Burning、Classic Pursuit、STYX HELIX、Blizzard Heights；请求 difficulty 4、原参考全曲 LN-head 比例，style 未指定。原生输出来自已完成的反馈对照；新增 16 份 source-H 输出只替换 head 时间，不提供源谱列、和弦、LN 类型或尾点。参考 H 均为整数毫秒，没有重新量化。

还核验了 2,506 张 metadata 难度位于 `[3.5,4.5)` 的 ranked 源谱及文件哈希。这个总体用于描述，不等于已分割的训练校准集，也不同于先前按本地难度实现重算的 TRAIN 总体。

测量区分以下量：

- **LN-head fraction**：新 LN 数 / 全部 head 数，不是占用时长。
- **Any-held fraction**：至少一列被持有的时间比例；高值可对应正常面条图。
- **LN duration**：真实尾点减起点；跨范围的 LN 不截短，未知尾点不伪造。
- **H span**：把真实 H 时间线线性映射到 H 序号后，LN 尾与头的坐标差。它描述跨过多少 H 间隔，不是 beat fraction。
- **相邻 LN 组长度变化**：按不同 LN onset 时间分组；对当前 LN，在前一组找最接近的长度，计算绝对 log 比值。前一组最多相隔 2 秒。分别在毫秒和 H 坐标计算，同时起始的长短 LN 另计组内差异。

后两项拆分 timing 与 release-span 的影响，**不以越小越好为目标**。Tech、不同声部和有组织的长短交替都可能有较大变化；前一组很丰富时，nearest-match 也会低估组织差异。

### LN 的对象是区间及其关系，不只是一个类型计数

一个 LN 是 $I_i=(s_i,e_i,k_i)$，分别为起点、终点和列。当前列占用是

$$
x_k(t)=\sum_{i:k_i=k}\mathbf 1\{s_i\le t<e_i\}.
$$

合法情况下同列这些区间不重叠。对 scope $W=[a,b)$，head 比例和占用是不同函数：

$$
\rho_W=\frac{\#\{i:s_i\in W,\ i\text{ 为 LN}\}}{\#\{\text{所有 heads in }W\}},
\qquad
u_W=\frac{1}{|W|}\int_W\sum_{k=1}^{4}x_k(t)\,dt.
$$

无 head 时 $\rho_W$ 未定义，而不是零。$u_W$ 包含从 scope 之前进入的 LN；积分也不要求在 scope 结束时释放。同样的 $\rho_W$ 可以对应截然不同的 $u_W$、释放频率、两手重叠与可恢复时间。因此用 LN 比例监督不能识别完整 LN 组织。

LN stream 的组织还在多个区间之间：entry 的次序、tail 与其他 entry 的相位、哪些列交接、同手是否同时承担持有和新攻击、关系是否在音乐重复时延续或变奏。这是一个带列标记的时间区间关系问题；只建模时长边缘分布 $p(e_i-s_i)$ 也会丢掉它。

### H 几何与持续决策的精确关系

对相邻 H 时间 $\tau^H_j,\tau^H_{j+1}$，定义仅用于分析的坐标

$$
\phi_H(t)=j+\frac{t-\tau^H_j}{\tau^H_{j+1}-\tau^H_j},\quad t\in[\tau^H_j,\tau^H_{j+1}].
$$

令 $\sigma_i=\phi_H(e_i)-\phi_H(s_i)$。当两端都在已观察的 H 范围内，LN 的毫秒长度精确满足

$$
e_i-s_i=\sum_j\omega_{ij}(\tau^H_{j+1}-\tau^H_j),\qquad
\omega_{ij}=\left|[\phi_H(s_i),\phi_H(e_i)]\cap[j,j+1]\right|.
$$

因此长度由 H 间隔和 LN 跨度／相位共同决定。所有 $\sigma_i=1$ 时，H 间隔从 `[100,100,100]` 变为 `[110,80,120]`，相同的“一间隔后释放”决策也会产生不同的实际节奏。这个关系不使用 redline，不要求任何固定 subdivision，也不排除高 fraction 采音或 dump。

相邻组描述量具体为

$$
v_i^{\rm ms}=\min_{j\in G_{\rm prev}}\left|\log\frac{e_i-s_i}{e_j-s_j}\right|,
\qquad
v_i^{H}=\min_{j\in G_{\rm prev}}\left|\log\frac{\sigma_i}{\sigma_j}\right|.
$$

未知尾点不进入上述完整长度比较；超出最后 H 的尾没有 $\sigma$，不会外推一个虚构 head。该分解解释来源，不定义“统一长度最优”。

## 3. LN 异常的具体定位

### 短 LN 本身不是异常

Ranked 总体中，108 张谱满足至少 50 个 LN 且有效谱面段 any-held fraction ≥ .75。其“每张谱 LN 长度中位数”的总体中位数是 **164ms**。参考 Blizzard 的 LN 长度中位数为 **89ms**，actor 与 memory 原生输出分别为 **171ms、101ms**。

因此“细碎”不能自动翻译成“长度太短”。需要解释的是短长关系、持续声部、释放与新按键的协调，以及这些关系如何连续发展。短 LN 比例只作诊断，不产生采样屏蔽。

### 多数被检查的短尾不是 deadline 强迫的

从生成行重建释放前的精确状态和 release window，区分 H 行、非条件化 R、deadline 条件化 R。条件化只表示某次释放必须在有限窗内发生，不等于恰好撞到强制 deadline。

| memory、seed 0、完整歌曲 | `<200ms` LN 数 | H 行结束 | 非条件化 R | 条件化 R | 命中 deadline |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classic Pursuit | 310 | 298 | 11 | 1 | 0 |
| Blizzard Heights | 1,060 | 917 | 120 | 23 | 1 |

Classic 的全部 H 释放行中，保持原行 head 数、列和 TAP/LN 决策，只把释放改成继续持有，仍满足现有短期支持检查；Blizzard 对应比例约 **99.88%**。这比“存在另一个同 head 数布局”更严格。

这排除了“恢复窗口把这些 H 行普遍逼成短尾”作为充分解释，却没有证明继续持有一定更好。支持检查只保证短期存在合法实现，好坏仍取决于后续释放、协调和压力。应学习这些可选持续／释放决策的后果，而不是放宽 floor 或把 LN 普遍拉长。

### H 的组织会传到 LN，R1 也有自己的偏好

仅替换为真实 source-H 后，memory Classic 的相邻组毫秒长度变化明显下降；在原生 H 上，其 H-span 变化中位数已经是零。重复相同的释放跨度，也会被不规则 H 间隔转换成不规则的实际 LN 长度。

下表是整曲相邻 LN 组绝对 log 长度变化的中位数，两个 seed 不合并：

| 端点／歌曲 | 原生 H，seed 0 / 1 | Source H，seed 0 / 1 | ranked 参考 |
| --- | --- | --- | --- |
| actor Classic | .495 / .479 | .425 / .439 | .0047 |
| memory Classic | .274 / .336 | .0093 / .0093 | .0047 |
| actor Blizzard | .358 / .363 | .172 / .094 | .0113 |
| memory Blizzard | .269 / .333 | .129 / .203 | .0113 |

H 是这些观测差异的一个原因，但干预同时改变间隔、数量与音乐对应，不能进一步声称已经单独定位为“毫秒 jitter”。这也不是部署修复，因为推理不能依赖源谱时间。

持续偏好差异仍存在：Classic 参考的 LN H-span 中位数为 **2**；memory 在原生 H 和 source H 下都为 **1**。更规整的 H 没有自动恢复该参考中的持续组织，actor Classic 的变化也有限。但源谱只是多种有效编排之一，不能单凭 1 对 2 的中位数判定 R1 错误；需要多参考分布与同前缀 continuation 响应来判断这种偏好是否不适合。

形式上，令 $G_\theta(A,c,T_H,\xi)$ 是给定 H 和随机种子后的完整生成，$M$ 为某一声明的观察量。source-H 对照测量

$$
\Delta M=M(G_\theta(A,c,T_H^{\rm ref},\xi))-M(G_\theta(A,c,T_H^{\rm native},\xi)).
$$

它是替换 H 的总体作用，包含 R/R1 因条件、到达状态和后续抽样位置改变而产生的中介作用。相同 seed 不意味着后续离散选择可以逐行相减；本对照不能识别“纯 timing 几何贡献”和“R1 重新响应贡献”各占百分之几。

18 页新 Lens 检查覆盖两个端点在 Classic `[88589,93589)` 和 Blizzard `[40342,56342)` 的 source-H 输出。memory Classic 的局部 LN/TAP 交织与节奏关系更清楚；actor Classic 仍在早段 LN 后转为较长 TAP 段。Blizzard 的 entry 更规整，但两端点仍以大面积持有为主，其持续／释放编排没有获得完整质量确认。高覆盖本身不是失败条件。第二个 seed 未做同等视觉检查；没有新的听音或玩家实测结论。

### 总量反馈不是充分解释

[LN feedback 对照](ln_feedback_scope_ablation.md) 已完成 36 份输出，18 对 H 完全相同。反馈额外偏好前缀比例均衡，真实且整体比例正确的纯 TAP 前段也可能把它推到上限。但移除反馈使 actor STYX 从约 .50 LN 比例变成约 .94，memory Classic 从约 .21 降到 .07，未形成跨歌曲的稳定组织改善。

需要检查 LN 类型分布、持续／释放分布、解析 ratio tilt 与训练策略之间的适应关系。只删除或加强反馈都没有获得本次证据支持；局部组织也不能通过全曲数量达标验收。

## 4. Long jack 的问题链与训练 caveat

[持续压力研究](time_horizon_player_responses.md) 记录了四星 Stream 回归：Zenithfall 某四秒段有 32 个 H、34 个 heads，列计数为 `[1,31,1,1]`，其他列没有被 LN 占用。同一 H 上较早 core 的计数为 `[11,14,10,14]`。该例不是 skeleton 强迫单指承担。

局部 recovery preference 在 difficulty 4 下，对超过约 107ms 的单次 HH 间隔不产生 HH rarity cost。连续数秒重复这种间隔，也不会由该项累积出独立 long-jack 负担。`frontier2` 可能从隐状态学到相关信息，但没有独立监督这种累计响应。

难度 proxy 也有漏洞：合成的 16 秒、每秒 12 个 H 的单列重复得到约 **4.141**，四列轮转为 **2.600**。它区分了两者，却仍可能让“达到 4 左右”的目标奖励难以接受的集中。这是 proxy 能力不足的反例，不是人类难度标注。

### 提案学习必须独立审查

源码及运行记录确认以下事实：

| 阶段 | 实际优化与覆盖 | 风险或未证明的性质 |
| --- | --- | --- |
| Scoped joint 基础训练 | 主要为 8 秒真源谱窗口；75% group/chart、25% human-chart 采样；音频编码冻结；恢复 profile 不兼容时重采样 | 接受分布不再严格 group-uniform；继承的预训练能力与新条件学习不能混为一谈 |
| `main-2000` 本阶段 | 接受 3,872 窗口、2,286 张谱；随后 500 步阶段为 1,000 窗口、787 张谱 | 不证明充分学习整个 corpus 或整曲状态；阶段之间谱数不可直接相加 |
| actor128 outcome | 47 个固定生成前缀场景，128 updates；每次 3 个真源谱 anchor、每请求 3 条独立续写；H/R 冻结 | 主要目标是难度标量和 LN 数量，缺少 LN 组织及累计协调响应；好标量可能由坏编排实现 |
| memory384 匹配实验 | 768 个曝光样本、417 张谱、383 song groups；全音频、H/R/R1 联合训练 | 有限学习对照；两个继续训练端点均有 native 退化，不能只归因于 outcome learning |

actor 的真源谱 anchors 与生成状态上的 outcome 分开，没有将生成前缀接源谱后缀冒充事实。问题在目标和状态覆盖仍不足。源谱项按时间归一化，outcome 的 score-function 项使用整段 chosen log-probability sum，乘固定权重 10。该求和符合轨迹梯度，不是数学错误；但权重的实际影响随事件数、horizon 和优势尺度变化，不能靠数字 10 判断广泛分布是否被保住。

memory 的已核验曝光只有 **6.368 秒**具有“只知道 Stream prominent、其他 style 未知”的条件。已知 style 范围为 .260–10 秒，生成请求可长达 144–358 秒；style dropout 主要按整组处理。独立可选 controls 的接口不等于训练覆盖了相应缺失模式。固定前缀 scope-clock probe 只产生较小 H 变化，因此该覆盖差距是待测 caveat，不能直接宣布为主要根因。

### 目标、采样分布与梯度的区别

完整 teacher-forced likelihood 的正确形式是

$$
\mathcal L_{\rm joint}=\mathbb E_{\mathcal D}\left[
-\log q^H_\theta(T_H\mid A,c)
-\sum_j\log q^{\rm wait}_\theta(e_j\mid\mathcal I_j,T_H)
-\sum_j\log q^1_\theta(a_j\mid\mathcal I_j,e_j,T_H)
\right].
$$

在足够数据、可实现模型和充分优化下，likelihood 学习真实联合分布具有统计依据。问题是有限训练、条件覆盖、模型归纳偏置和部署策略使这些前提未被满足；“NLL 是 proxy”不等于应放弃或错误实现联合概率。

恢复约束还改变有效训练总体。若原采样分布为 $p_{\rm draw}(g,w\mid b)$，$g$ 是 song group／chart，$w$ 是窗口，$b$ 是 population 或 human 分支，接受条件为 $A_\kappa(g,w)$，则保持分支不变重采样得到

$$
p_{\rm accepted}(g,w\mid b)
=\frac{p_{\rm draw}(g,w\mid b)\mathbf 1[A_\kappa(g,w)]}
{\Pr_{p_{\rm draw}}(A_\kappa\mid b)}.
$$

这不是原始 group-uniform 分布。必须记录实际曝光和拒绝，不仅记录采样意图；“某张谱存在一个被排除关系”也不等于整张谱的所有窗口被拒绝。

actor 的 outcome 实验还固定了源谱 H，并验证生成未改变这些 H。因此它优化的任务更接近

$$
J_B(\theta)=\mathbb E_{(h,T_H^{\rm ref},c)\sim B}
\mathbb E_{Y\sim\pi_\theta(\cdot\mid h,T_H^{\rm ref},A,c)}
\left[L_D(\widehat D(h\oplus Y),c)+L_{\rm LN}(\rho(Y),c)\right].
$$

部署则同时涉及生成 H、从 BOS 到达的状态分布，以及尚未被 $L_D,L_{\rm LN}$ 充分表达的响应。即使 $J_B$ 得到正确梯度，也没有自动优化整个部署任务。改善旧前缀的续写不保证新策略不会制造新的坏前缀；只在 authored H 上学会响应也不保证对 generated H 鲁棒。

若只更新 R1 参数，冻结音频/H/R，对实际轨迹的 score-function 梯度可以只包含 R1 项：

这里假定该更新步骤的响应损失没有显式 $\theta$ 依赖，baseline 对本次抽样动作独立；若响应模型共同更新，须分清其显式梯度和策略梯度。

$$
\nabla_\theta\mathbb E_{\pi_\theta}[L]
=\mathbb E_{\pi_\theta}\left[(L-b)\sum_j\nabla_\theta\log\pi^1_\theta(a_j\mid h_j)\right].
$$

R 的实际结果仍会随先前 R1 改变的状态而变化，这个间接作用已经由轨迹分布承担，不要求对冻结 R 虚构一个参数梯度。反之，共享音频／H/R 参数参与更新时，不能沿用“只记行概率”而遗漏这些参数实际进入的随机因子。baseline $b$、得分 horizon 和控制 scope 也必须符合该估计的条件。

### 音频能到达 readout，不代表能改变所需的相对偏好

一个已确认的结构例子是未启用 layout modulation 的主行头。两手输入为 $z_L=h_L+d(A,T_H,c)$、$z_R=h_R+d(A,T_H,c)$，其完整候选 score 对这两个输入是仿射的：

$$
J(z_L,z_R)=L_Lz_L+L_Rz_R+b.
$$

由于左右镜像等变和共享加法，对候选 $a$ 及其反射 $Ma$ 有

$$
\frac{\partial}{\partial d}\left[J_a(h_L+d,h_R+d)-J_{Ma}(h_L+d,h_R+d)\right]=0.
$$

因此这条主路径不能让音乐／style 根据当前历史改变“继续原组还是换到镜像组”的相对偏好。相同 count-family 的 composition 与 LN amount feedback 也在该比较中抵消。非线性 routing、frontier 和已实现的乘性 modulation 可以打破限制，所以这不是完整模型对音频盲目的证明。详见 [条件—历史交互的推导及对照](row_condition_interactions.md)。actor 已启用 modulation，故不能把其剩余失败再次简单归因于缺少该路径。

该恒等式固定历史及支持，只针对共同加法条件进入主 readout 的路径。额外 history reader 也可能通过改变 $h_L,h_R$ 引入条件交互，完整 rollout 还能经先前动作产生间接影响；这些都须保留在实际架构审计中。

这个例子支持一种研究方法：先检查需要表达的条件交互是否能改变相关候选的 odds，再决定是否加网络、换接口或扩大容量。它比“已有 audio embedding／attention，所以信息充分”更具体。

## 5. 架构应该如何按职责调整

需要保留 H/R/R1 的决策所有权，并把**提案概率、玩家状态、未来响应、选择／提交**分开。

### 条件提案模型

提案学习音频与控制下合法、多样的编排分布。音频必须直接进入行编排；只有 beat 或 skeleton 不足以表达音色、层次与音乐重复。各因子可以共享音频表示，也可以增加为关系建模服务的模块。

待比较的两个假设是：H 是否需要更明确的跨事件节奏关系／共享局部音乐变量；R1/R 是否需要可持续追踪的 LN 起点音乐信息和 entry–continuation–release 关系。年龄、TCN 和 attention 理论上可能学到它们，现有实验没有证明已经学会。新增表示应提高不同后果的可分辨性，而不只是增加 token。

H 神经网络完全不读已提交动作摘要，是研究选择而非 V3 必然。有限、声明清楚的玩家响应摘要可以成为比较方向；不能因此把 head 数／布局转移给 H，或重新允许无法解释的 generic R1 hidden state 支配所有 timing。

### 玩家状态与 continuation response

以精确 replay 和按真实时间保留的历史为基底，响应表示需要保留这些区别：

- 一列持续重复，与总量相同但负担转移的攻击历史；
- 正在持续、刚进入、刚释放的 LN，以及释放后衔接其他动作；
- 两手／同手动作的相位、重叠和协调，而不只是四个独立 rate；
- 释放后的真实恢复，与没有新 head 但仍在按住的时间。

这些是需要校准的对比，不是预先规定的生理曲线。ranked 难度与 scoped style 提供证据，不能单凭相关性把隐变量命名为疲劳。

响应算子读取同一历史状态和**实际私有续写**，在数秒到十余秒的真实 horizon 上比较 entry、release、coordination、jack 累积与恢复。可先建立保留丰富历史的参考预测器，再压缩成在线 state；是否足够由响应比较决定。

控制目标用于解释和比较响应，不重置状态。音乐兼容性仍由提案／独立音乐判断处理；corpus 风格分类器不能替代玩家响应。`RowConsequence` 可保留为快速提案偏置，但不应被当成已校准的完整 frontier。

### 状态充分性是关于未来响应的命题

令 $d_\psi(h_t)$ 为候选玩家状态。所需近似是

$$
\widehat{\mathcal C}_\psi(x(h_t),d_\psi(h_t);Y,e)
\approx \mathcal C_0(h_t,t;Y,e),
\quad (Y,e)\in\mathfrak T,
$$

$\mathfrak T$ 是声明的合法 continuation／horizon 家族。精度只能针对这个家族、这些响应通道和独立证据来讨论，不能因某个 hidden size 足够大就宣称充分。

若两段历史满足 $x(h)=x(h')$、$d_\psi(h)=d_\psi(h')$，同一预测器必然对相同未来给出相同输出。若目标响应需要区分它们，该状态就不充分。一个应检验的构造是：相同 H 和总 head 数，较早几秒分别为单指集中与多指转移，随后使用共同短后缀使最近各列攻击时刻和占用相同，再要求继续攻击原过载列。精确状态可以相同，但按所需累计响应语义，未来负担不应只由最后一个间隔决定。当前 TCN 可能保留这种区别；实验要检查是否保留且被正确使用，不能仅凭输入名推断。

一种可解释的候选基底是按真实事件时间过滤，而不是固定 row 数：

$$
d^{\rm press}_{k,r}(t)=\sum_{u\in\mathrm{press}_k,\,u\le t}K_r(t-u),
\qquad
d^{\rm release}_{k,r}(t)=\sum_{v\in\mathrm{release}_k,\,v\le t}K_r(t-v),
$$

$$
d^{\rm hold}_{k,r}(t)=\int_{-\infty}^{t}K_r(t-u)x_k(u)\,du.
$$

例如 $K_r(\Delta)=\exp(-\Delta/\tau_r)$、$\tau_r>0$ 可以提供多个时间尺度；$\tau_r$ 和后续读出需校准。以毫秒为时钟时，前两个量是加权事件数，第三个量是加权持有毫秒，不能不加定义直接求和当作 strain。它们也不能独自表达动作顺序和两手协调，需配合保留顺序、时差和占用的 $z_t^{\rm coord}$。该组合是待比较的表示选择，现有 box-window reference 是另一种基底；不因公式看起来像衰减就认定符合人类响应。

动态实现至少保持以下 invariant：同一前缀 replay 与 cache 一致；无新行的时间推进满足分段一致性；控制更新不清除状态；私有未来只更新各自的副本。固定精确占用条件下，分段一致性可写为

$$
V_{\Delta_2}\circ V_{\Delta_1}=V_{\Delta_1+\Delta_2}.
$$

开着的 LN 仍持续输入 hold 通道，不能因“没有事件”而被当作全手休息。这些 invariant 保证计算含义一致，响应是否正确仍需外部校准。

### Planner 与 scheduler

Planner 从同一已提交边界展开候选、调用响应算子、选择未来；scheduler 管理预算、时限、缓存与不可撤销前缀。固定 H 的原型有利于隔离 R/R1，却无法修复所有 timing 问题。没有合适的 R/R1 续写时，可以比较未提交的 H 候选，同时保留已发布行依赖的短期可行性承诺。

预算耗尽后的最小 cost 不自动等于合格。应能报告“尚未找到合适未来”，再比较更好的 proposal、响应或私有搜索范围；不新增短 LN、某种 jack、不均匀分指的形态禁令。现有恢复 profile 的支持排除也需继续量化，不能代替质量判断。

### 提案覆盖与选择失误可以分别量化

对给定 $A,c,h_t,e$，用 $\mathcal G$ 表示独立评价认为合适的未来集合。这是分析记号，不是实现一个手写形态 mask；未解决的语义判断保持未知。令有限候选集为 $\mathcal Y_K$，选中结果 $Y^*\in\mathcal Y_K$。则

$$
P_{\rm success}
=\Pr(\mathcal Y_K\cap\mathcal G\ne\varnothing)
\Pr(Y^*\in\mathcal G\mid\mathcal Y_K\cap\mathcal G\ne\varnothing).
$$

第一项是 proposal coverage，第二项是 conditional selection accuracy。若独立同分布提案的单次合适概率为 $p$，第一项为 $1-(1-p)^K$；真实共享／自适应搜索应直接测量，不能擅用独立假设。$p$ 极小时仅增加 $K$ 的效率很差，而 coverage 高、selection 低时扩大生成模型也未必解决问题。

对每个独立声明的响应损失 $L_r$，可以记录候选内 regret

$$
\operatorname{Regret}_r=L_r(Y^*)-\min_{Y\in\mathcal Y_K}L_r(Y).
$$

不能把不同控制区间和响应通道任意相加后抵消，也不能用待训练选择器自己的损失作为唯一独立评价。特别要保留“最佳候选仍然差”和“好候选存在但选错”的区别。

经过 planner 选择的分布 $\pi_{\rm selected}$ 也不同于单条 native proposal $q$。例如在连续成本、无 ties 的理想情况中，固定 $K$ 个独立提案、按固定成本 $L$ 取最小值时，

$$
p_{\rm selected}(Y)=Kq(Y)\,[1-F_L(L(Y))]^{K-1}.
$$

其中 $F_L$ 是 proposal 成本的分布函数。实际自适应 planner 不必满足这个简化形式，但它揭示了训练风险：不能把选中轨迹的 native log-probability 当成 selected-policy likelihood。学习 proposer、蒸馏 selected continuations 和对搜索策略求梯度，是三个不同的训练问题。

### 实时性约束的是计算策略，不应偷偷改变质量定义

设当前已发布覆盖比播放位置领先 $B_i$ 秒，下一次计算耗时 $\tau_i$，新发布长度为 $\Delta_i$。保持不中断至少要求在 buffer 用尽前完成，更新后

$$
\tau_i\le B_i,\qquad B_{i+1}=B_i-\tau_i+\Delta_i.
$$

搜索预算 $K_i$、forecast horizon、共享音频缓存和响应近似可以据此调度。增加参数或训练预算可以接受，但必须重新测首窗、dense passages 和最坏 service，而不只报告整曲平均速度。临时缩减搜索不应重写已提交状态、伪造 LN 尾或将未通过响应检查的结果标成合格。候选不足时需要明确的系统降级契约；当前尚未建立完整保证。

实时基线还需对齐 `codex/stream-generation-benchmark`，本次读取的提交为 `4ec631ef71d1ca71e36e5c383d4997efffdebb05`，说明文件是该 ref 的 `docs/research/controlled_audio_runtime_benchmark.md`。其中旧 aligned 端点的 90 次整曲、3,060 个八秒发布窗口给出 service p95 .300s、最大 .474s，resident readiness 最大 .589s。这说明存在工程余量，但不是新 memory、frontier 搜索或未来大模型的延迟保证。

该 benchmark 的 readiness 是至少 30 个物理行与 8 秒已确认覆盖；本研究 qualifier 使用另外声明的两秒 lead／service 条件，两者不能混用。后续修改至少应在 benchmark 所属流程回归以下契约：

- producer watermark 代表已经交付完整事件的覆盖，不是请求的时间、最后一个 note 时间或已有 H preview；静默已完成与尚未算完不同。
- Mel／完整音频编码可以按 frontend、checkpoint、device、dtype 等身份共享；replay、LN feedback、H 队列、各 RNG、survival residual 与历史／玩家状态由每个 session 或私有分支独立拥有。
- 普通发布边界允许开放 LN，真正 EOS 才要求音频结束和合法闭合；控制更新、有限读取、stall、取消和长 LN 都要保持不可撤销前缀。

该分支报告的是生成／预处理测量和部分虚拟 buffer 回放，并未建立完整传输／客户端保证。本报告没有重新运行该分支，也没有改动其 worktree；架构扩展必须把这条真实运行路径作为后续回归对象，而不只测试一个离线生成函数。

## 6. 后续实验应回答的明确问题

### A. 好候选是否存在，还是选择器没选到

固定从 BOS 原生到达的前缀，覆盖 Classic／Blizzard 的 LN 问题、STYX 的真实复杂释放组织、Zenithfall／Hysteric 的 long jack，并保留 ranked 合理重复反例。在同音频、控制和 H 下，展开相同预算的多条 R/R1 私有未来。

分别记录合适候选出现率、最佳可见候选、实际选择与差距。依据来自冻结的独立诊断和已读 Lens witness，不能用待训练 frontier 自身分数定义“好候选”。供给不足先改提案；候选存在却排名错先改响应。固定前缀组是机制／回归工具，训练仍需刷新当前策略从 BOS 到达的状态。

### B. 更广的真实分布是否被 recipe 保住

同一父模型、相近算力和一致 native panel 下，比较真源谱继续训练、改善数据／control 缺失模式覆盖的训练、再加入完整 continuation-response 目标。保留同音频多种真实编排，不合并成帧级并集。

核验实际接受曝光，而不只看 sampler 配置：song groups、难度、LN 覆盖与持续关系、style 已知／未知组合、控制范围长度、被 profile 排除的窗口。真源谱 imitation 使用真实前缀，生成状态学习使用自身真实后果；分开记录梯度贡献、源谱能力保留与 native 分布漂移。

容量、attention 或 timing 表示也可以进入这个匹配比较，不必先证明小模型绝对不够。需要明确每项改变预期修复的能力，以及实际数据／目标是否同时变化；允许为互相依赖的模块联合设计，但不能用规模遮盖目标不充分或分布过窄。

### C. 玩家响应能否区分“总量相近、后果不同”

构建总 head 数／LN 比例相近的对照：集中与转移攻击、持续与碎裂持有、规则 LN stream 与不易跟随的 entry/release 组合、不同占用下的相同未来。既包含生成失败，也包含真实 ranked 难例与正常变体。

响应必须对实际 horizon 和历史敏感，控制切换时状态连续。目标可以是有证据的相对偏好、多通道响应或校准分布，不能只拟合原有星级 proxy。未经实测／独立标注支持的量保持为经验响应，不宣称测得人类生理上限。

## 7. 可复用评估与验收状态

[LN timing relations](../../src/ensomi_model/research/gameplay_evaluation/hold_relations.py) 已接入 [scope report](../../src/ensomi_model/research/gameplay_evaluation/report.py)，报告真实时间与 H 坐标关系、组内差异、censored holds 和最大变化上下文。构造回归验证：相同 H-span 的释放选择在 H 时间扰动后，毫秒关系改变而 H-span 关系不变；局部范围不伪造尾点、不丢失前驱上下文。

它仍是诊断，不能自动判断 LN stream 好坏。持续压力、占用、恢复、多尺度变化、音乐对应和发布时限独立报告。评估应允许高覆盖面条、Tech、LN coordination 和合理 jack，不奖励全部变稀或变整齐。

[Native qualification](gameplay_regression_evaluation.md#executable-native-qualification) 实际运行了 16 份 source-H 输出。memory 数值条件全部通过，状态仍是 `review_required`；actor 两个 Blizzard 输出的全曲 LN 数量条件失败。两者均未晋级。完整 source-H 诊断用时 122.01 秒；它是有参考时间输入的机制实验，不是可部署音频生成的质量通过。

全曲 LN 请求被后续 override 打断后，前段和恢复后的片段不自动变成新的总量请求。各段分别报告，只对完整声明范围检查总量，避免评估重新强迫前缀均衡。

这一区分可以写成：scope 总量要求限制 $\rho_W$，不自动限制每个 $\rho_{[a,t)}$。当前投影反馈

$$
b_{n+1}=\operatorname{clip}\left(b_n+\frac{\rho h_n-\ell_n}{8},-2,2\right)
$$

在请求 $\rho=.5$ 时，32 个 TAP head 后已到 $+2$；64 TAP 后再 64 LN 的整体正确序列最终还会到 $-2$，因为 clipping 丢弃了累积债务。它引入了额外前缀偏好。替代方案需要保持 scope 控制而允许音乐决定局部分配，不能仅把这个偏好移到 evaluator 里。

## 已有证据对应的假设与下一步决策

下面是可被实验削弱或支持的假设，不是已经选择的模型补丁。它们共同服务于提案覆盖和玩家响应两个条件，不能把某一条的局部改善当成整套系统完成。

| 假设 | 已有支持及限制 | 最能改变判断的比较 | 结果应影响的设计 |
| --- | --- | --- | --- |
| H 的节奏关系／条件分布偏移是 LN 异常的重要来源 | Source-H 改善部分关系，但同时改变数量、音乐对应和到达状态；actor Classic 仍有显著差异 | 在固定预算下比较 native-H 与 authored-H 条件下合适 R/R1 候选的供给，再检查相近 H 密度下的节奏关系 | H 表示、关系建模或 training recipe；不能把 R1 的 counts 转给 H |
| 持续／释放选择缺少对 LN 关系的建模 | 许多 H 行可以保持原布局并继续持有；H-span 偏好仍不同于参考，但这不等于每次 release 都错 | 同一真实到达前缀，对合法 hold/release 分支展开实际未来，比较占用、entry/release 协调及独立阅读结果 | LN origin／持续关系表示、R/R1 联合提案学习和 continuation response |
| 窄条件曝光和 outcome 目标使提案分布偏移 | 47 个固定前缀、source H、标量目标与稀少的 Stream-only 曝光；source-only 也会退化 | 同父模型、同算力、匹配源谱样本，分别改变曝光／缺失模式和 outcome 目标，保留 native BOS 检查 | Recipe、保真约束、目标通道和当前策略状态刷新；不能先认定仅 RL 有问题 |
| 当前响应表示／监督不足，导致已有好候选未被接受 | 本地 frontier 没有实际未来输入；planner 不含 LN／协调响应 | 用同一候选集比较原选择、独立评价的最佳候选和 richer-history 响应模型 | 独立响应学习、状态充分性、搜索／提交职责；只加输入而不改目标未必有效 |
| 模型容量限制了广泛真实分布与音乐关系 | 有合理动机，但现有更大 memory 端点未通过 native 质量 | 明确记录目标／覆盖，用同一独立 evaluator 做匹配的容量／交互比较，区分必要的联合修改与无关差异 | 可以扩大网络、attention 或训练预算；以 native coverage／selection 和时限决定，而非参数量直觉 |

下一步优先建立可比较的候选供给与选择证据。围绕明确机制，可以合并相互依赖的架构与 recipe 修改，通过匹配对照保留可归因性。约束是证据、语义和实时契约可追踪，不是维持现有模块数量或参数规模。

## 8. 版本与证据身份

Native 实验 source：`24061b918be7075f1efc70341675fe066c2e45b6`；Apple M5、24 GiB、Torch 2.11.0、CPU 单线程，没有并发训练。新 LN 关系评估是后续只读分析，不改变输出。所有实验已结束，本分析没有产生新训练端点。

| 证据 | SHA-256 |
| --- | --- |
| actor128 | `364c7b711bc39ab3384175b755658d2ef67a4c10b166f5eb85dc072d920e10e3` |
| memory384 | `7efcbe4bd6da3a4ec9ead01f24fdda63803ea930ebe76bdc7781fbf080a97e81` |
| Ranked 总体冻结记录 | `bd7685153cdf0aab98bd2dc70e7063a31775db49db919b4b1114a003b1227c28` |
| 2,506 张谱描述记录 | `d8388e2fd7a7ef7782916f32e1dc67604b620294e63d42bca459f10f0bbe799c` |
| 36 份原生输出的释放来源 | `6c2f00740a099514e72bd05aa4bca9a382071054e79bb52cdbc1f9eb9139e95c` |
| Source-H 计划 | `799664d00b9d3df568c30246e74b020ac11791a183496e6dd397901af8b1abb9` |
| Source-H 运行记录 | `a66a892062a8d7cb29db07f7fdc6af5c81b287345e73b89daf0ba8df2ea045c2` |
| 时间关系／保持 LN 反事实分析身份 | `c051e8e1d2b92c15fbb67d071ee36812102e754d5cb9195aadad5032a0f82506` |

Lens harness：`22e5c84f5cacb8493bdab5f1d0fdc09c5373dc60`。实验标识 `20260927-ln-continuation-diagnosis-v1` 的 `run-v2`、`source-head-v2`、`decomposition-v1`、`lens-source-head` 保存原始记录。checkpoint、音频、谱面和生成资产不包含在普通 fresh clone 中；本文的结论、测量定义和后续问题不依赖本地文件才能理解。
