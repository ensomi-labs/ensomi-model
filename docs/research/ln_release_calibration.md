# LN 释放的条件校准：总似然改善可以掩盖什么

在固定的真实谱面历史上，完整行似然的改善可以同时伴随释放子决策的退化。
这不是“所有 LN 都太短”的单向偏置，也不能靠把所有尾巴统一延后解决。
需要区分：新 head 的类型／位置、已有 LN 的继续／结束、多个手指是否共同释放，以及上游 R 是否已经决定发生一次释放。

本研究新增了[条件释放评估器](../../src/ensomi_model/research/gameplay_evaluation/release_calibration.py)，
并对[干净联合训练](clean_joint_proposal_learning.md)的三个初始化及 32／512 步终点进行事实历史诊断。
它没有改变生成器或定义新的 BAD 阈值，也不是一个已经校准的玩家 frontier。

## 测量对象

所有模型看到同一真实 prefix、源谱 H 预看、完整音频及控制；真实 LN 尾巴只作为后续观察标签，不进入当前查询。
控制使用源谱整谱星级与整谱 LN-head fraction，style 未指定。
概率是实际研究采样 law：60/25/21 ms 支持、direct LN conditioning、LN feedback 关闭，保留既有 empirical recovery preference。

每个 H 行前已存在的 LN 构成一个 release risk。对列 $i$，从完整行分布求

$$
p_{i,t}=\sum_{a:\,a_i=\mathrm{release}}q_{\rm R1}(a\mid \xi_t).
$$

$\xi_t$ 是当前事实条件，观测 $y_{i,t}$ 表示该 LN 是否真的在此行释放。
保留 age、过去一秒的 H 数、其他持有列，以及有限支持中是否存在继续持有的选择。
H 与纯 R 分开统计；R 行上的 mark 概率以释放时间已被选中为条件，不评估 R 的时间分布。

完整 release subset 的概率为

$$
p_t(S)=\sum_{a:\,\{i:a_i=\mathrm{release}\}=S}q_{\rm R1}(a\mid\xi_t),
\qquad S\subseteq\{0,1,2,3\}.
$$

同时记录单列、成对共同释放、完整 subset 的 Brier 误差。
例如，两指可以各有 .5 的 release marginal，但联合分布分别是
“一起结束／一起继续各 .5”与“只结束其中一指各 .5”。
前者给共同结束 .5 概率，后者给零；单列均值无法区分它们。
这些是预测关系，不是额外的物理动作计数或压力单位。

对一个实际仍保持的 LN，还计算它在所观察 H 查询上的继续概率乘积。
该乘积沿事实 prefix 计算；它不是自由生成的寿命概率，因为替代动作会改变之后的状态和 R 时刻。
进入窗口的 LN 保留原始起点，未结束者保持删失。无 H 查询时不虚构一个生存估计。

参考动作必须独立于被评估策略的采样。对模型自己生成的动作做校准，只能描述决策如何发生，
不能证明生成质量。选中的八个 TRAIN 片段也不构成总体校准检验：
Brier 是条件拟合诊断，参考谱只是一个有效编排，并不要求所有真实选择都得到概率一。

## 固定真实上下文

这些来源已在[ranked 协调对照](coordination_frontier_hypotheses_zh.md)中完整阅读。
时间为原始毫秒、半开区间；不删除短 LN、不吸附尾巴。

| 来源 | Beatmap ID | 范围 | 整谱星级 |
| --- | ---: | --- | ---: |
| Shizuku [Marble soda] | 4185651 | 120000–128000 | 4.4491 |
| The Last Page [Grand Finale] | 2697513 | 40000–48000 | 4.5173 |
| Non-breath oblige | 4148629 | 32000–40000 | 4.3446 |
| Someone In The Crowd [Soulmate] | 2910941 | 240000–248000 | 4.4467 |
| Cafe + M!lk + Chocolate [Hot Chocolate {Insane}] | 3880856 | 64000–72000 | 3.9036 |
| Until the end of time [Himitsu's Hyper] | 3937519 | 72000–80000 | 3.6435 |
| Kanzen ShouriEsper Girl [Esper] | 4144572 | 104000–112000 | 3.8219 |
| Bedroom community [Eternal Slumber] | 3036683 | 232000–240000 | 3.8466 |

它们包含大量共同释放，也包含合法短 LN trains、独立 release 与持续 anchor 的反例。
这些正反组织都需要保留；“更多共同释放”不是统一的优化方向。
所有 factual target row 都在被评估模型支持内，没有通过排除不合适标签制造结果。

## 不是一个全局 release 偏置

初始 inherited 模型在 Shizuku 的 97 个 H／held-column risk 上预测
57.94 次释放，实际为 65；在 The Last Page 的 51 个 risk 上却预测
33.61 次，实际为 28。Hot Chocolate 为 23.15 对 17，Esper 为 32.55 对 35。
一个统一偏置不能同时校准这些片段。

Shizuku 的分解更具体：实际继续的 32 个 risk 上，模型分配的释放概率之和为
17.09；实际结束的 65 个 risk 上为 40.85。共同结束的 pair 实际有 22 次，
模型期望为 9.84。总量偏少与在某些持有位置分配较高结束概率可以同时出现。
这说明应检查概率落在哪些关系上，不能从一个平均寿命推断结束策略已正确。

源谱在 121880／121973 ms 分批按下的列 2／1，于 122067 ms 共同结束；
后面的列 1／2 在 122161／122255 ms 开始，于 122348 ms 共同结束。
在 122255 ms，列 1 仍应作为该真实动作组的一部分继续，模型 marginal release
概率为 .7411；在另一个 93 ms age 的继续位置，概率仅 .1864。
网络并非完全不读上下文；问题不能简化为“年龄超过某值就结束”的唯一机制。

## 把 head 选择与 release 选择拆开

Marginal release 概率还混合了模型提出的不同新 head 配置。
进一步定义 $u(a)$：保留完整行中的 TAP／LN press 及其列，将 release 替换为空动作；
剩余释放集合为 $v(a)$。这两个变量都是 R1 内部的完整行决定，$u$ 不是 timing skeleton。

对任意已归一化行 law，有精确分解

$$
\begin{aligned}
q_U(u\mid\xi)&=\sum_{a:u(a)=u}q_{\rm R1}(a\mid\xi),\\
q_V(v\mid u,\xi)&=q_{\rm R1}(a(u,v)\mid\xi)/q_U(u\mid\xi),\\
-\log q_{\rm R1}(a\mid\xi)
&=-\log q_U(u(a)\mid\xi)-\log q_V(v(a)\mid u(a),\xi).
\end{aligned}
$$

第二个诊断固定**当前观察到的 head signature**，检查 release subset。
这是条件概率分解，不把真实当前 head 作为部署时额外可用的输入。
纯 R 行只有空 head signature，所以其 head 项为零。
全部查询的数值分解已检查到 1e-10 容差；原始 marginal law 与该条件 law 分开保存。

以下为各自八秒范围内的 NLL 总和，单位 nats。这里只比较同片段、同控制的初始与 512 步端点：

| 模型与片段 | Head signature NLL | Release given heads NLL | 完整行 NLL |
| --- | ---: | ---: | ---: |
| Early，Shizuku，初始 → 512 | 101.37 → 85.81 | 62.86 → 74.73 | 164.23 → 160.54 |
| Inherited，Non-breath oblige，初始 → 512 | 138.30 → 105.98 | 72.85 → 82.19 | 211.15 → 188.17 |
| Fresh，Until the end of time，初始 → 512 | 166.20 → 160.53 | 75.85 → 118.68 | 242.05 → 279.21 |

前两例直接证明：**完整行 NLL 下降，也可以掩盖条件释放项变差。**
第三例说明 fresh 的整体验证进步并非所有 LN 组织都改善；
它不是从头初始化已经失败的结论，512 步仍是早期端点。

Until the end of time 中，列 1 从 72035 ms 持有到 72635 ms，
列 3 在其间完成 75 ms 的短 LN 序列。固定真实当前 heads 后，
fresh-512 对列 1 在 72185／72335／72485 ms 结束的概率为
.7878／.7480／.7224；同样条件下 inherited-512 为 .1859／.1480／.1216。
这定位到“另一指继续短动作时，是否保留当前 anchor”的条件选择。
未训练 fresh 在这些条件下的结束概率很低，但对应 head signature 的概率也仅
.0103／.0132／.0193；不能把一个罕见条件分支上的偶然结果当作已学会该组织。

## 对训练与架构的含义

首先，需要分别保留 timing H、timing R、R1 head signature 与 release subset 的学习证据。
这些分解不替代完整生成，但可以防止一个容易改善的部分掩盖另一个退化的部分。

一个可检验的训练分支是

$$
L=L_H+L_R+L_U+\lambda_V L_{V\mid U},\qquad\lambda_V>0.
$$

$\lambda_V=1$ 恢复原来的联合行似然。对无限表达能力的条件分布，
超额期望风险为

$$
\mathrm{KL}(p_U\|q_U)
+\lambda_V\,\mathbb E_{p_U}
\mathrm{KL}(p_{V\mid U}\|q_{V\mid U}),
$$

所以正权重不改变理想的数据分布最优点；它改变有限模型中的学习权衡。
这比把所有 LN 拉长更直接，也不需要把稀有 LN 谱当成未知控制的默认先验。
但共享参数仍可能改变 head、音频与后续到达状态，必须保留实际 native 与 control guards。
本研究没有启动该加权分支，也没有宣布它能修复可玩性。

另一个表示假设是保留 LN 出生时的编排／音乐上下文，帮助区分持续角色与短动作组。
那属于 proposal 的编排记忆；不能把一个私有“预计何时结束”的意图当作玩家已经经历的事实。
玩家 demand state 仍只能由已提交动作构成，frontier 仍要评价完整候选未来。

最后，条件拟合只是“能否学到这种真实关系”的证据之一。
它不能确定不同合法替代的玩家困难顺序，更不能决定完整生成是否可玩。
继承模型在这些事实状态上有较好的局部能力，仍可能从 BOS 进入错误的 TAP／LN 模式；
native H 也可能使低星目标在任何 R1 编排下不可达。此类状态访问、时间支持与选择问题须另行检查。

## 实现与证据身份

评估器代码提交为 `ef90a21943ed4b56070f2679c3d3c9a2883be699`；
模型源为 `ef42095a6e764b0374edbaa36b8ddf87c32364c9`。
评估器在独立 worktree 开发，没有改变正在运行的主训练代码。
九项聚焦测试覆盖同 marginal／异 joint、上下文与删失、范围可加性、
镜像变换、纯 R 的强制性、原子交接、时钟对齐和浮点归一化。

本地 owner 为 `artifacts/joint-audio/20260928-ln-risk-calibration-v1`。
首轮八上下文 × 六模型状态完成于 19.18 s；扩展到初始／32／512、
加入条件 head 分解后，八上下文 × 九模型状态完成于 42.86 s，
最大采样进程 footprint 为 777,733,536 bytes。均为 CPU 单线程，没有额外训练。
这些选例结果是探索性机制诊断，不是总体校准证明、独立人类标签或 playtest。

| 证据 | SHA-256 |
| --- | --- |
| 固定来源与范围 | `ce40ee32c6a98c8a5834640f4b2df7562939ba6d56276911a11c81b78d683ba3` |
| 首轮结果 | `29a2e1a97eeca51109a203d012661979b42c877e0fef2d00f2b2d82406fd9ef2` |
| 扩展执行计划 | `f234a12a9439eb2b3d3f9a565df7e36e06a773e75493be27c87fa71749903be0` |
| 扩展结果 | `1f80b8e24b1d3ea9c62d26c77210864e99034f4af3373f29d7cea8d7bfd31de1` |
| 扩展逐例与概率分解 | `013c4512171f2e57e5023dbe05c914f7403735092fb68a3101fe8eb74ea11e64` |
