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

## 进一步区分放几个与放谁

令 $K=|V|$ 为释放数量，则同一完整行 law 还可以精确写成

$$
-\log q(a\mid\xi)
=-\log q_U(u\mid\xi)
-\log q_K(k\mid u,\xi)
-\log q_V(v\mid u,k,\xi).
$$

这不是在 R1 前增加一个强制释放计划；它是对已经存在的联合 law 做诊断。
[row_likelihood_parts](../../src/ensomi_model/research/gameplay_evaluation/row_likelihood.py)
返回每个查询的三个 NLL 项和固定 $(U,K)$ 后仍有多少有限支持的身份选择。
只有一个可选子集时，identity NLL 必然为零，不能当作已学会协调的证据。
真实源行不在支持内时，条件分解会明确失败，不把未定义条件伪装成零损失。

同样的八秒真实上下文给出：

| 模型与片段，初始 → 512 | Release count given heads NLL | Release identity given heads/count NLL |
| --- | ---: | ---: |
| Early，Shizuku | 56.938 → 69.034 | 5.921 → 5.697 |
| Inherited，Non-breath oblige | 51.922 → 63.106 | 20.928 → 19.086 |
| Fresh，Until the end of time | 43.158 → 79.419 | 32.695 → 39.258 |

前两例的条件释放退化来自“放几个”项；给定数量后选择具体手指反而略有改善。
这比“联合 release 学坏了”更精确。第三例两项都变差。
数值仍是同一真实编排在当前 law 下的预测误差，不是玩家需求单位。

Until 的 72260 ms 是不同的问题：应释放列 3 的短 LN，继续列 1 的 anchor。
Inherited-512 给“只放一指” .902659 的概率，但在这个条件下，
只有 .269213 分配给列 3，其余分配给 anchor。
而在 72185／72335／72485 ms，真实动作应保留唯一进入中的 LN；
这里条件 identity 没有选择，过早结束完全体现为 cardinality 错误。
从玩家视角看，同一个持续角色会在两类决策中被破坏：不该增加一次放键，
以及确实要放键时却打断了另一指的持有。短 LN train 本身不是错误标签。

### 诊断因子不等于神经模块

当前 [RowComposition](../../src/ensomi_model/research/controlled_audio_continuation/model.py)
预测的 mark 是

$$
m(a)=(n_{\rm head}(a),n_{\rm LN\ head}(a),n_{\rm release}(a)).
$$

因此固定 $U$ 仍未固定 mark：release count 会变化。
对本文模型，令 $\ell(a)$ 为 layout 分数、$\pi_m$ 为 count 概率、
$Z_m=\sum_{a':m(a')=m}\exp\ell(a')$ 为合法组内归一化量、
$g(a)$ 为 learned row consequence、$c(a)$ 为采样恢复偏好，则

$$
q(a)\ \propto\
\exp\!\left[\ell(a)-\log Z_{m(a)}+\log\pi_{m(a)}+g(a)-c(a)\right].
$$

只有同时固定 $(U,K)$，$\log\pi_m-\log Z_m$ 才抵消：

$$
q_V(v\mid u,k)\ \propto\
\exp\!\left[\ell(a(u,v))+g(a(u,v))-c(a(u,v))\right],
\quad |v|=k.
$$

所以 count 分支不直接改变这个条件下的手指身份 odds；
但 $q_K(k\mid U)$ 也不等于 count 网络的一个独立输出，
它还包含具体 head 配置的 layout 质量与后加能量。
不能把上表的 cardinality 退化直接归罪于 count 网络的某组权重。
共享音频与历史参数还会跨这些诊断因子产生训练影响。

## 当前行后果模块在补偿与加重什么

在相同真实 prefix、音频、控制、H/R 查询和有限支持下，
对八个上下文、九个模型状态比较四个 law：
当前部署 law、只去掉 $g$、只去掉 $c$、两者都去掉。
捕获 compose 和 consequence 的实际分数后，可重建原 law；
这里没有重新训练、重新生成历史或改变上游时钟。

| 512 步片段 | 部署 $L_{K\mid U}$ | 去掉 $g$ 后 | 部署 $L_{V\mid U,K}$ | 去掉 $g$ 后 |
| --- | ---: | ---: | ---: | ---: |
| Early，Shizuku | 69.034 | 75.264 | 5.697 | 5.612 |
| Inherited，Non-breath oblige | 63.106 | 67.707 | 19.086 | 18.689 |
| Inherited，Until the end of time | 21.718 | 21.593 | 23.822 | 25.252 |
| Fresh，Until the end of time | 79.419 | 75.698 | 39.258 | 40.276 |

行后果能量在前两例补偿了一部分 cardinality 错误。
在 inherited Until 的整段上，它改善了 identity 项；
但在单个 72260 ms 时刻，去掉它使正确手指的条件概率从 .2692 升到约 .3370，
仍偏向错误 anchor。整体帮助与局部加重可以同时成立。
这些结果不支持把该模块统一删除，也不把它证明成玩家响应模型。

在 Shizuku、Non-breath 和 Until 的这些真实查询中，
移除 empirical recovery preference 不改变条件 cardinality／identity 损失，
数值差小于 1e-13；它不能解释这里的释放误差。
它在其他来源上确实有影响：例如 Bedroom 的部分模型／role 分量变化可达 4.265 nats。
因此这个排除只适用于所述事实状态，不能推广为偏好对 native 历史无影响。

固定 law 的能量拆解也不是历史训练责任的唯一分解。
不同网络分支在共同 NLL 下可以相互补偿；删掉一项后的误差，
不等于“从头不训练该项”所得模型的误差。

## 对训练与架构的含义

首先，需要分别保留 timing H、timing R、R1 head signature、release count 与 release identity 的学习证据。
这些分解不替代完整生成，但可以防止一个容易改善的部分掩盖另一个退化的部分。

一个可检验的训练分支是

$$
L=L_H+L_R+L_U+\lambda_K L_{K\mid U}
  +\lambda_I L_{V\mid U,K},\qquad\lambda_K,\lambda_I>0.
$$

$\lambda_K=\lambda_I=1$ 恢复原来的联合行似然；两者相等时是对整个条件 release 项加权。
固定条件 $\xi$，记真实联合行为 $p$、模型为 $q$。对无限表达能力的条件分布，
仅行项的超额期望风险为

$$
\mathrm{KL}(p_U\|q_U)
+\lambda_K\,\mathbb E_{p_U}
\mathrm{KL}(p_{K\mid U}\|q_{K\mid U})
+\lambda_I\,\mathbb E_{p_{U,K}}
\mathrm{KL}(p_{V\mid U,K}\|q_{V\mid U,K}),
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

最初条件释放评估器的代码提交为 `ef90a21943ed4b56070f2679c3d3c9a2883be699`；
模型源为 `ef42095a6e764b0374edbaa36b8ddf87c32364c9`。
评估器在独立 worktree 开发，没有改变正在运行的主训练代码。
条件释放评估器的九项聚焦测试覆盖同 marginal／异 joint、上下文与删失、
范围可加性、镜像变换、纯 R 的强制性、原子交接、时钟对齐和浮点归一化。
三因子评估器另有六项测试，覆盖准确条件分解、count 权重不能改变固定数量的身份 odds、
镜像、极小概率、singleton／空查询与不支持源行的区别。
两个评估器的 15 项聚焦测试全部通过；三因子实现另与保存的 288 个 law、
23,616 个查询逐项比较，最大差异为浮点归一化产生的 3.25e-7 nats。

本地 owner 为 `artifacts/joint-audio/20260928-ln-risk-calibration-v1`。
首轮八上下文 × 六模型状态完成于 19.18 s；扩展到初始／32／512、
加入条件 head 分解后，八上下文 × 九模型状态完成于 42.86 s，
最大采样进程 footprint 为 777,733,536 bytes。均为 CPU 单线程，没有额外训练。
这些选例结果是探索性机制诊断，不是总体校准证明、独立人类标签或 playtest。

保存概率的 cardinality／identity 分解耗时 .180 s、最大 RSS 32,063,488 bytes，
链式分解的最大误差为 4.27e-14 nats。四个 law 的固定前缀比较耗时 27.86 s，
最大采样 footprint 875,923,016 bytes，输出约 21.65 MB；
该执行使用只增加本文初版文档的 evaluator revision
`5b0dbeb6a5c5c5240bde0cf8325e54fee365d6c0`，模型源未变。

| 证据 | SHA-256 |
| --- | --- |
| 固定来源与范围 | `ce40ee32c6a98c8a5834640f4b2df7562939ba6d56276911a11c81b78d683ba3` |
| 首轮结果 | `29a2e1a97eeca51109a203d012661979b42c877e0fef2d00f2b2d82406fd9ef2` |
| 扩展执行计划 | `f234a12a9439eb2b3d3f9a565df7e36e06a773e75493be27c87fa71749903be0` |
| 扩展结果 | `1f80b8e24b1d3ea9c62d26c77210864e99034f4af3373f29d7cea8d7bfd31de1` |
| 扩展逐例与概率分解 | `013c4512171f2e57e5023dbe05c914f7403735092fb68a3101fe8eb74ea11e64` |
| Cardinality／identity 分解 | `0821580897f76e57b6f771dccb80349f18913dc3c3845cffc96d438072218c30` |
| 固定前缀能量拆解计划 | `b76a6df9a5a36fadfbf54ac3ebee9a1b77871148c3970a61d0280b139b03858b` |
| 固定前缀能量拆解结果 | `1522b41d640b99424d4e667a6c720dd471f042e5bb98278fb7e0cee004f18f6c` |
| 能量拆解逐例 | `9a85f7a14baf5858109ca114177613f485deb8843d8a70eac3793f85e6ff7036` |
