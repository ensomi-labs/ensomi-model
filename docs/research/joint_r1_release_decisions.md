# 在选定释放时间之前，让 R1 决定保持还是结束

独立 R 时钟先决定“这里发生释放”，R1 再选择释放子集，会让部分保持决策发生得太晚。
只有一个可释放 LN 时，任何有限的行偏好在单一 mark 的归一化后都会消失。
另一条结束路径却一直存在：在 H 时刻，R1 可以主动结束 LN。
因此把所有短尾归因于 R，或只延后 R，都不足以解决实际生成。

实现增加了 `release_policy="r1_joint"` 研究模式。
非 H 时刻由完整 R1 对“继续等待／释放各子集”共同评分，R 的 hazard 从这个联合分布计算。
H 时刻仍由 R1 决定完整行，H 本身仍只给出 head-bearing 时刻。
旧的独立 R 模式保留为对照；新模式尚未通过真实训练及可玩性验收。

## 实际短尾由谁决定

下表重放了[干净联合训练](clean_joint_proposal_learning.md)的 inherited-2048 输出。
“短”在此固定为持续时间 ≤80 ms 的描述坐标，不是生成的硬性最短时长。
两个请求均为 D4；Stream 没有指定 LN amount，STYX 指定整曲 LN fraction 约 .485。

| 观察 | Stream Zenithfall | STYX |
| --- | ---: | ---: |
| 全部 LN | 4,382 | 758 |
| ≤80 ms LN | 2,670，60.93% | 84，11.08% |
| 短尾发生于 H | 2,524 | 13 |
| 短尾发生于纯 R | 146 | 71 |
| 纯 R 短尾中，进入时只有一条 LN | 110 | 26 |
| 短尾命中非末尾强制 deadline | 0 | 0 |

所有这些 H 上短尾，在保持当前 head signature 不变时都存在继续持有的合法行。
因此 Stream 中 94.53% 的短尾是行策略在 H 上可改变的决定；
STYX 的近端责任却更多落在纯 R 时间。
这只是当前支持下的决策归属，不把早先的 H 密度、LN birth 选择和训练过程排除在因果链之外。

一种必要的整体检查来自持有库存：长期平均占用与 LN birth rate、平均持续时间相关。
在相同高 birth rate 下单纯延长尾巴，会减少可活动手指，可能把压力挤到少数手指；
硬 H 计划又可能迫使某些 LN 提前结束。改动必须同时检查同列压力、LN 数量、占用和 H 节奏。

## 已固定的红灯回归

[short_ln_exposure](../../src/ensomi_model/research/gameplay_evaluation/ln_fragmentation.py)
计算一个完整 scope 中

$$
B_{80}(Y)=
\frac{\#\{\text{LN heads with actual duration}\le80\text{ ms}\}}
{\#\{\text{all heads}\}}.
$$

参考来自[1,973 张 3.5–4.5★ ranked 谱](ordinary_fourstar_rhythm_and_holds_zh.md)，
每个歌曲组等权、组内谱等权。明确 LN request 选择其 source amount 层；
未指定则使用自然混合，不允许生成结果自行选择较宽松的参照。
未指定 amount 的 99 分位是 **.20014**。

固定的 inherited／early Stream 输出分别为 **.55245／.43933**，均失败。
16 张源谱 sanity controls 全部通过，包含普通长持有、短 LN、LN 主体和非二进制节奏来源。
这是对已知失败建立的回归，不是盲测，也不是把 80 ms 写入解码器。
其余 native checks 保留：少生成 LN 或统一拉长尾巴不能单凭这一坐标获得模型晋级。

## 共同的等待与释放分布

在一个尚未决定是否发生 R 的 native-ms 时刻 $t$，令 $\mathcal M_t$ 是可行的非空释放子集，
并加入虚拟动作 $\varnothing$ 表示这一时刻没有物理行。
R1 使用实际行历史、精确占用／时钟、完整音频、H preview 和 controls，
为这 16 种至多四键的 wait/release 动作给出分数 $s_\theta(a\mid\xi_t)$。
可选的已有 LN-origin audio cues 仍只读取真实起点，不读取未来尾巴。

令 $\eta$ 为 flow log scale，初始化为 $\log .001$，对应将秒尺度的相对 odds 投到 native-ms 查询；
它是可学习的强度坐标，不是最短 LN 时长。定义

$$
\begin{aligned}
Z_t&=\exp s_\theta(\varnothing)+
 \exp\eta\sum_{M\in\mathcal M_t}\exp s_\theta(M),\\
p_t(\varnothing)&=\exp s_\theta(\varnothing)/Z_t,\\
p_t(M)&=\exp[\eta+s_\theta(M)]/Z_t,\\
\operatorname{logit}h_R(t)&=
\eta+\log\sum_{M\in\mathcal M_t}\exp s_\theta(M)
-s_\theta(\varnothing).
\end{aligned}
$$

R 的事件概率为 $h_R=1-p_t(\varnothing)$，发生事件后的 mark law 为
$p_t(M\mid M\ne\varnothing)$。同一组分数因此同时影响“此刻是否放”和“具体放谁”。
不是先固定 R，再用一个无法拒绝事件的行评分补救。

在一个单 LN 状态，即使条件 mark 概率恒为一，提高 wait 分数仍会降低事件 hazard。
同样，作用于这个 release 的 soft recovery cost 现在改变事件 odds；
它不再只是在单一 mark 的归一化中消失。

对当前允许的 flat/no-count-prior/no-allocation 模式，
将 all-empty 加入纯 R 支持，只新增一个 count group；
非空行之间的相对 odds 与现有 R1 条件 mark law 一致。
实现对这项等价、梯度及实际 replay 作了检查，不要求近似独立地训练一个 mark surrogate。

## 时钟、约束与训练

虚拟 wait 不进入 row cache、skeleton cache 或玩家状态，也不会发给客户端。
调度器只提交抽到的非空行，保留 no-event coverage 和指数采样的剩余阈值。
在必须腾出按键的 deadline 或真实 audio end，仍有明确的强制释放 atom。
此前发布的行、未结束 LN 和控制切换语义保持原有协议。

新模式保留原始 survival law，到最后一个可行时刻才施加强制 atom：

$$
\Pr(T=t,M)=
\left[\prod_{u<t}(1-h_R(u))\right]h_R(t)\,
p_t(M\mid M\ne\varnothing),\qquad h_R(d)=1.
$$

它不再把整段 wait 重新归一为“已知某次释放必定在 deadline 前发生”的条件时钟。
否则在低强度近似下，整体降低 release odds 可能被条件归一化抵消。
这也允许按有限时间块评分，不必先计算一直到歌曲末尾的全部假想 wait。
强制 atom 仍只是物理可行性，不能证明该 H 计划或 LN birth 合理。

训练的时间项加上非空 mark 项，正是上述 joint wait/mark likelihood；
H 上完整行损失保持其原有意义。
旧 independent release MLP 在新模式下不再参与 R 的计算；
release 梯度改为进入 R1、共享音频和相关条件路径。
旧参数保留在兼容 checkpoint 中，不把它们的存在计作已参与学习的能力。

数据与调用必须匹配：

```python
batch = collate_interval(
    example, model.config, device, recovery=model.recovery,
    release_policy=model.release_policy,
)
encoded = model.encode_audio(complete_song_mel, real_frame_mask)
scores = score_interval(
    model, batch.inputs, None, controls=controls, encoded_full=encoded,
    recovery_preference=the_sampling_preference,
)
```

两种执行都使用完整音频。source H／历史的 teacher forcing 仍是训练的条件分解；
这没有消除 source-prefix 与 native 到达状态之间的差距。
训练时没有额外的真实 BPM／phase 输入，也没有把未来 LN 尾巴作为当前观测。

## 当前实现范围与下一步

纯释放查询只评分 16 个候选，避免把 256 行的中间激活全部保存。
它与 dense scoring 的值和梯度一致。已检查 native hazard 与当前权重 replay、
publication partition、跨 control 边界、deadline atom、checkpoint reload 和 CPU/MPS 梯度。
这些是概率与执行证据，不是训练后的质量证据。

这次实现尚未重构 H 的节奏层级。后续 H 必须让难度在给定局部音乐速度下影响主 subdivision，
并允许更细修饰；不要求先得到唯一且无误的 beat/BPM 表。
内部时间单位或相位候选应从音频及 note-placement 目标共同学习，
避免修饰事件把主节奏参照一起重置。R1 仍负责 chord、分指、LN 类型与保持／结束。

在系统验收前，还需完成有界补训、固定红灯及完整 native guards、真实谱面对照，
以及 `codex/stream-generation-benchmark` 的实际首窗／密集服务测试。
这个研究模式没有自动晋级。
