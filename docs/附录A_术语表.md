# 附录 A　术语表

> 术语按主题分组，组内大致按学习顺序排。每条给出中文名、英文、一个简短的定义，以及第一次详细讲它的章节。中文一栏用的是正文里的叫法；有些概念正文直接用英文（如 all-reduce、rollout），这时中文一栏是帮助理解的译名，正文里搜英文才能找到。英文一栏是你在论文和代码里会看到的写法。

---

<a id="sa-1" name="sa-1"></a>
## A.1 全流程与算力

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 预训练 | pre-training | 在万亿级无标注文本上做下一个 token 预测，所有 token 都算 loss | 1 |
| 中期训练 | mid-training | 预训练之后、SFT 之前再训一段，目标仍是下一个 token 预测，但换成更高价值的数据，并拉长上下文 | 1、16 |
| 继续预训练 | continued pre-training | 从已有权重出发，目标不变，换数据接着训 | 1、16 |
| 监督微调 | SFT, supervised fine-tuning | 用对话数据训练，只对助手回复算 loss | 1、17 |
| 强化学习 | RL, reinforcement learning | 模型自己生成回答，按奖励调整生成概率 | 1、18 |
| 基座模型 | base model | 只做过预训练（和 mid-train）的模型，会续写，不会按指令回答 | 1 |
| 检查点 | checkpoint | 训练某一步保存下来的状态，至少含权重 | 1、13 |
| dense 模型 | dense model | 每个 token 都经过全部参数的模型，和 MoE 相对 | 1 |
| 混合专家 | MoE, mixture of experts | 把 FFN 换成多个并列的"专家"子网络，每个 token 只激活其中一部分，所以总参数多、每 token 的计算少 | 1、9 |
| 浮点运算次数 | FLOPs | 训练或推理做的乘法和加法总次数 | 1 |
| 6ND | — | 训练总 FLOPs 的估算：$N$ 是参数量，每 token 前向约 $2N$、反向约 $4N$，再乘训练 token 数 $D$ | 1 |
| 每秒浮点运算 | FLOP/s | 硬件或训练实际达到的运算速度；注意和 FLOPs（总次数）区分 | 1 |
| 模型算力利用率 | MFU, model FLOPs utilization | 每秒实际完成的模型 FLOPs 除以硬件峰值 FLOP/s，衡量卡被用满了多少 | 1 |
| 到达同一 loss 的时间 | time-to-loss | 训练到某个 loss 要花的时间，由吞吐和样本效率共同决定 | 1 |
| 样本效率 | sample efficiency | 每个 token 能让 loss 降多少 | 1 |
| 全局 batch 大小 | GBS, global batch size | 一次参数更新用到的序列条数 | 1、12 |
| 步 | step / iteration | 一次参数更新 | 1 |
| 算力最优比例 | Chinchilla scaling | 固定算力下 loss 最低的参数量与 token 数之比，约每参数 20 token | 7 |

<a id="sa-2" name="sa-2"></a>
## A.2 模型结构

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 仅解码器 Transformer | decoder-only Transformer | 只用解码器层堆起来、每个位置只能看前文的 Transformer，GPT、LLaMA、ZGCM 都是 | 2 |
| 嵌入 | embedding | 按 token 编号查表得到向量 | 2 |
| 隐藏维 | hidden size, $d$ | 每个位置的向量长度，ZGCM 是 4096 | 2 |
| 注意力 | attention | 每个位置拿自己的查询向量 q 和各位置的键向量 k 打分，再按分数对各位置的值向量 v 加权求和 | 2 |
| 因果 mask | causal mask | 把"未来"位置的打分屏蔽掉，让每个位置只能看到自己和之前的位置 | 2 |
| 多头注意力 | MHA, multi-head attention | 把向量拆成多个头，每个头用自己的 q、k、v 各做一遍注意力，结果拼回去 | 2 |
| 分组查询注意力 | GQA, grouped-query attention | 多个 Q 头共享一组 K、V，KV cache 随组数变小；ZGCM 是 32 个 Q 头配 8 组 | 2 |
| 多查询注意力 | MQA, multi-query attention | 所有 Q 头共享一组 K、V | 9 |
| KV 缓存 | KV cache | 推理时把已生成位置的 K、V 存下来，避免重算 | 2、9 |
| 旋转位置编码 | RoPE, rotary position embedding | 把 q、k 按位置旋转一个角度，两者点积只依赖相对距离 | 2、9 |
| RoPE 基数 | RoPE base, $\theta$ | 决定各维旋转频率的常数，越大转得越慢 | 9、16 |
| 部分旋转 | partial RoPE | 只对每个头的一部分维度做旋转，其余维度不带位置信息；ZGCM 转 128 维里的 42 维 | 9 |
| QK 归一化 | QK-Norm | 对每个头的 q、k 先做 RMSNorm，限制它们的长度，从而给注意力打分设上界 | 2、9 |
| 注意力输出门 | attention output gate | 注意力输出逐元素乘一个 0 到 1 之间的 sigmoid 门，决定放行多少；ZGCM 只在滑窗层用 | 2、9 |
| 前馈网络 | FFN, feed-forward network | 每层的第二个子层，对每个位置单独做的两层或三层 MLP | 2 |
| SwiGLU | SwiGLU | 一种 FFN 写法：两路投影，一路过 SiLU 后与另一路逐元素相乘，再投影回去 | 2 |
| RMS 归一化 | RMSNorm | 只按均方根缩放、不减均值的归一化 | 2 |
| 残差连接 | residual connection | 子层输出加回输入 | 2 |
| 前置归一化 | Pre-LN | 先归一化子层输入，再计算，再加回残差 | 2 |
| 后置归一化 | Post-LN | 加完残差后再归一化，原始 Transformer 的做法 | 2 |
| 输出层 | lm_head | 把最后的隐藏向量投影成词表大小的打分 | 2 |
| 对数几率 | logits | 输出层给词表里每个 token 的原始打分，过 softmax 才变成概率 | 2 |
| 不共享嵌入 | untied embedding | 输入嵌入表和 lm_head 是两套参数 | 2、9 |
| 滑动窗口注意力 | SWA, sliding window attention | 每个位置只看最近 $w$ 个位置，ZGCM 的 $w=128$ | 9 |
| 全局层 | global (full) attention layer | 看全部前文的注意力层，ZGCM 有 5 层 | 9 |
| 多头潜在注意力 | MLA, multi-head latent attention | DeepSeek 的做法，缓存一个低维潜在向量代替 K、V | 9 |
| 线性注意力 | linear attention | 把前文压进一个固定大小、逐步更新的状态，代替 softmax 注意力，计算随长度线性增长 | 9 |
| 初始化标准差 | init std | 权重初始化正态分布的标准差，Megatron 默认 0.02 | 9 |

<a id="sa-3" name="sa-3"></a>
## A.3 Loss 与评测指标

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 交叉熵 | cross-entropy, CE | 正确 token 概率的负对数，语言模型的 loss | 2 |
| 困惑度 | perplexity, PPL | $e^{\text{loss}}$，可理解为"平均在几个候选里犹豫" | 2 |
| 每字节比特数 | BPB, bits per byte | 把 loss 换算成每个 UTF-8 字节要多少比特，不同 tokenizer 可比 | 2、15 |
| 少样本 | few-shot | 题目前放几道带答案的例题，让模型照着格式答 | 15 |
| 选项字母打分 | MC, multiple-choice formulation | 把选项标成 A/B/C/D 写进题目，比较模型给这几个字母的概率 | 15 |
| 选项原文打分 | RC, rank classification / cloze | 把每个选项原文分别接在题后，比较（长度归一化后的）对数概率，不要求模型懂 ABCD 格式 | 15 |
| 标准误 | SE, standard error | 同一个模型换一批同类题再测，分数会上下浮动，标准误是这个浮动的典型幅度；正确率的标准误是 $\sqrt{p(1-p)/n}$ | 14、15 |
| 置信区间 | confidence interval | 按标准误给出的真实值可能范围，95% 约为 ±1.96 SE | 19 |
| 自助法 | bootstrap | 有放回地重抽题目，重算分数，用分布估区间 | 19 |
| 配对比较 | paired test | 两个模型（或检查点）答同一组题，只数两边对错不同的题来估差值的噪声，比把两次分数当成独立的估得更准 | 15 |
| pass@1 取 k 次平均 | avg@k, mean pass@1 | 每题采样 $k$ 次，正确率取平均 | 19 |
| pass@k | pass@k | 采 $k$ 个回答至少一个对的概率；实际采 $n$ 个、其中 $c$ 个对，用无偏估计 $1-\binom{n-c}{k}/\binom{n}{k}$ | 19 |
| 多数投票 | maj@k | 采 $k$ 个回答，取出现最多的答案 | 19 |
| 饱和基准 | saturated benchmark | 大家都接近满分、区分不出强弱的基准 | 19 |
| 地板基准 | floor benchmark | 大家都接近零分的基准 | 19 |
| in-loop 评测 | in-loop evaluation | 训练过程中定期在检查点上跑的小规模评测 | 15 |
| 短 SFT 探针 | short-SFT probe | 给基座做一次快速 SFT 再评测，估它变成助手后的水平 | 15 |
| 模型裁判 | LLM-as-judge | 用另一个模型判断答案对错 | 19 |
| 评测框架 | evaluation harness | 负责提示格式、调用模型、抽取答案、打分的代码 | 15、19 |

<a id="sa-4" name="sa-4"></a>
## A.4 Tokenizer

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 分词器 | tokenizer | 把文本切成 token 并映射成整数编号 | 3 |
| 子词 | subword | 介于字和词之间的切分单位：常见词整个是一个 token，少见的词拆成几块 | 3 |
| 字节对编码 | BPE, byte-pair encoding | 反复把最常一起出现的相邻两块合并成新块，得到词表 | 3 |
| 字节级 BPE | byte-level BPE | 从 256 个字节起步的 BPE，任何文本都能切，不会碰到词表里没有的字符（没有未登录词） | 3 |
| 预切分 | pre-tokenization | BPE 之前先用正则按单词、数字、标点、空白切开，合并不跨边界 | 3 |
| 词表大小 | vocabulary size, $V$ | token 种类数，ZGCM 的 tokenizer 是 154,856 | 3 |
| 词表填充 | vocab padding | 把嵌入表行数补到某个倍数，便于张量并行（TP）均分和内存对齐 | 3 |
| 特殊 token | special token | 不来自 BPE 合并、有特殊用途的 token | 3 |
| 文档结束符 | EOD, `<\|endoftext\|>` | 每篇文档末尾加的 token，标记文档边界 | 3、8 |
| 对话模板 | chat template | 把多轮对话拼成一串 token 的规则，含角色标记 | 3、17 |
| 压缩率 | compression ratio | 每个 token 平均对应多少字符或字节 | 3 |

<a id="sa-5" name="sa-5"></a>
## A.5 数据：来源与清洗

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| Common Crawl | Common Crawl | 每月抓取一次全网的公开网页存档 | 4 |
| WARC | WARC | Common Crawl 存原始 HTTP 响应的文件格式 | 4 |
| 正文抽取 | text extraction | 从 HTML 去掉导航、广告、模板，留下正文 | 4 |
| 光学字符识别 | OCR | 从扫描件或 PDF 图片里识别文字 | 4 |
| 只发索引 | index_only | 不发原文，只发去哪找原文和原文哈希的记录 | 4 |
| 出处 | provenance | 每条记录的来源、版本、许可证等元数据 | 4 |
| 启发式过滤 | heuristic filtering | 用词数、重复行比例、标点结尾比例等规则丢文档 | 4、5 |
| Unicode 规整 | NFKC normalization | 把全角、兼容字符等统一成标准形式 | 5 |
| 三档决策 | keep / review / drop | 保留、待复核、丢弃，ZGCM 清洗流水线的判决 | 5 |
| 数据集 profile | profile | 针对某个上游数据集的一组专属清洗步骤 | 5 |
| 个人身份信息 | PII, personally identifiable information | 邮箱、电话、身份证号等能认出具体个人的信息 | 5 |
| 密钥扫描 | secret scanning | 找出文本里的 API key、私钥等凭据 | 5 |
| 质量分 | quality score | 上游或自训分类器给每篇文档的质量打分 | 5、6 |

<a id="sa-6" name="sa-6"></a>
## A.6 数据：去重、质量、去污染

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 精确去重 | exact dedup | 内容（或规整后的内容）哈希相同就算重复，只留一份 | 5、6 |
| 内容哈希 | SHA-256 等 | 把任意长文本映射成定长摘要，改一个字摘要就完全不同 | 5、6 |
| 近似去重 | near dedup | 找出"几乎一样"的文档，比如只差导航栏或日期 | 6 |
| 瓦片 | shingle / n-gram | 把文本切成连续 $n$ 个词（或字）的片段，作为比较的特征 | 6 |
| SimHash | SimHash | 把文档的所有特征哈希后按位投票，压成一个 64 位指纹；相似文档指纹相近 | 6 |
| 汉明距离 | Hamming distance | 两个指纹有几位不同；ZGCM 仓库的阈值是 3 | 6 |
| 分段索引 | band index | 把 64 位指纹切成 4 段 16 位分别建索引，只和至少一段相同的旧指纹比 | 6 |
| 鸽笼原理 | pigeonhole principle | 把东西放进比它少的格子里，必有一格放了不止一个。用在分段索引上：最多差 3 位、分 4 段，3 处不同最多落进 3 段，至少有 1 段完全相同，所以不会漏 | 6 |
| Jaccard 相似度 | Jaccard similarity | 两个集合交集大小除以并集大小 | 6 |
| MinHash | MinHash | 用 $k$ 个哈希函数各取集合里的最小值做签名，两篇文档签名相同的比例可以估计它们的 Jaccard 相似度 | 6 |
| 局部敏感哈希 | LSH, locality-sensitive hashing | 把签名分 $b$ 段、每段 $r$ 个值，有一段全同才进候选；阈值约 $(1/b)^{1/r}$ | 6 |
| 子串去重 | exact substring dedup | 用后缀数组找出语料里重复出现的长片段，只删片段 | 6 |
| 后缀数组 | suffix array | 把一个长串的所有后缀按字典序排好的索引，重复子串会相邻 | 6 |
| Bloom 过滤器 | Bloom filter | 位数组加 $k$ 个哈希的集合结构，判"没见过"一定对，判"见过"可能误报 | 6 |
| 跨阶段去重 | cross-stage dedup | 新阶段的数据和之前阶段已用过的数据之间去重 | 6 |
| 质量分类器 | quality classifier | 学一个模型给文档打"值不值得学"的分，按阈值筛 | 6 |
| FineWeb-Edu 分类器 | FineWeb-Edu classifier | 用 70B 模型给约 45 万篇网页的教育价值打 0–5 分，再训一个小模型给全量打分，留 3 分以上 | 6 |
| 评测污染 | contamination | 评测题或答案出现在训练数据里 | 6 |
| 去污染 | decontamination | 训练前删掉和评测集重合太多的样本；ZGCM 的 SFT 用 8-gram、重合超过 50% | 6、17 |

<a id="sa-7" name="sa-7"></a>
## A.7 数据：配比与课程

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 数据配比 | data mixture | 每类数据在训练 token 里各占多少 | 7 |
| 自然比例 | natural proportion | 按每类数据现有的量直接混 | 7 |
| 温度采样 | temperature sampling | 把每类数据的量开 $\alpha$ 次方（$0<\alpha\le1$）再归一化当作采样比例，放大小类、压缩大类 | 7 |
| 代理模型 | proxy model | 用来做数据决策的小模型，ZGCM 是 0.3B、训约 30B token | 7 |
| RegMix | RegMix | 用很多小模型跑随机配比，拟合"配比→loss"的回归，再找最优配比 | 7 |
| DoReMi | DoReMi | 训小代理模型时，哪类数据的 loss 比参考模型高得多就调大它的权重，把学到的权重拿去训大模型 | 7 |
| 课程学习 | curriculum learning | 训练数据按难度等指标排顺序，而不是随机 | 7 |
| 词汇复杂度 | lexical complexity | ZGCM stage 1 给通用文本排序用的指标，报告没给公式 | 7 |
| 全局打乱 | global shuffle | 所有来源的样本混在一起打乱，ZGCM stage 2 在写索引文件前离线完成 | 7、8 |
| 遗忘 | forgetting | 换数据继续训后，旧能力变差 | 7 |
| 回放 | replay | 新阶段数据里混入一部分旧阶段数据，防遗忘 | 7、16 |
| 轮次 | epoch | 一份数据完整过一遍 | 7 |
| 数据重复 | data repetition | 同一份数据训多遍；约 4 遍以内和等量新数据差不多，再多收益快速下降 | 7 |
| 长度分桶 | length bucket | 按文档长度分组采样，控制长文的占比；见 `prefix_sample_mix` | 7、16 |

<a id="sa-8" name="sa-8"></a>
## A.8 数据：从文本到训练样本

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 索引数据集 | IndexedDataset, `.bin` / `.idx` | Megatron 的磁盘格式：`.bin` 连续存 token，`.idx` 记每篇文档的起点和长度 | 8 |
| 内存映射 | mmap | 把文件映射进地址空间，用到哪页读哪页，不必整个装进内存 | 8 |
| 文档顺序表 | document index | 一维数组，列出按什么顺序读文档；数据要过几遍，每篇文档就在里面出现几次 | 8 |
| 样本切分表 | sample index | 第 $j$ 条样本从哪篇文档的哪个偏移开始 | 8 |
| 样本顺序表 | shuffle index | 第几次取样本时取切分表里的哪一条 | 8 |
| 打包 | packing | 把多篇文档首尾相接，切成等长样本，不留填充 | 8 |
| 跨文档注意力 | cross-document attention | packing 后不加掩码，后一篇文档能看到前一篇；ZGCM 预训练就是这样 | 8 |
| 位置编号重置 | reset position ids | 每篇文档的位置从 0 重新数；ZGCM 预训练没开 | 8 |
| 累积序列长度 | cu_seqlens | 一个 pack 里各段文档的起止位置，变长注意力算子靠它让注意力不跨段 | 8、17 |
| 混合数据集 | BlendedDataset | 按权重把多个来源合成一个样本流；每次看哪个来源已出的样本比目标份额落后最多（"最大缺额"），就从它取下一条 | 8 |
| 已消费样本数 | consumed_samples | checkpoint 里记的"已经训过多少条样本"，续训从这里接着读 | 8、13 |
| 样本偏移 | sample offset | 换数据续训时从 consumed_samples 里减掉的数，让新数据从第 0 条读起 | 8、13 |
| 顺序读取 | sequential sampler | 不再打乱，按样本顺序表原样读，保留课程顺序 | 8 |

<a id="sa-9" name="sa-9"></a>
## A.9 优化器与学习率

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 学习率 | LR, learning rate | 每步更新的步长系数 | 10 |
| 随机梯度下降 | SGD | 参数沿负梯度方向走一步 | 10 |
| 动量 | momentum | 用梯度的指数移动平均代替当步梯度，减少抖动 | 10 |
| Adam | Adam | 用梯度的移动平均 $m$ 定方向，再除以梯度平方的移动平均的平方根 $\sqrt v$，每个元素有自己的步长 | 10 |
| 偏差修正 | bias correction | $m$、$v$ 从 0 起步会偏小，除以 $1-\beta^t$ 补回来 | 10 |
| $\beta_1$、$\beta_2$ | betas | $m$、$v$ 的衰减系数，ZGCM 是 0.9 和 0.95 | 10 |
| 权重衰减 | weight decay | 每步把参数按比例往 0 拉一点，ZGCM 是 0.1 | 10 |
| AdamW | AdamW | 把权重衰减从梯度里拆出来、直接作用在参数上的 Adam | 10 |
| 奇异值分解 | SVD | 把矩阵拆成两组正交方向和每个方向的放大倍数（奇异值） | 10 |
| Muon | Muon | 对二维权重的动量矩阵做近似正交化（奇异值拉到约 1），让各方向的更新幅度大致相同，再用它更新 | 10 |
| Newton–Schulz 迭代 | Newton–Schulz iteration | 只用矩阵乘做近似正交化的多项式迭代，ZGCM 用 5 步 quintic（五次多项式）系数 | 10 |
| spectral 缩放 | spectral scaling | 正交化后的更新乘 $\sqrt{\max(m,n)}$，让每元素改动约等于 LR，不同形状的层可比 | 10 |
| 走 Adam 的参数 | scalar params, embedding, lm_head | 一维参数（如 RMSNorm 增益）加 embedding 和 lm_head，ZGCM 不用 Muon，约占参数的 17% | 10 |
| 梯度范数 | grad norm | 所有梯度拼成一个向量后的 L2 长度 | 10、14 |
| 梯度裁剪 | gradient clipping | grad norm 超过阈值就整体缩到阈值，ZGCM 是 1.0 | 10 |
| 预热 | warmup | 开头若干步把 LR 从 0 线性升到目标值，ZGCM 预训练是 667 步 | 10 |
| 常数调度 | constant schedule | warmup 之后 LR 不变，ZGCM 预训练就是这样 | 10 |
| 余弦调度 | cosine schedule | LR 按余弦曲线从峰值降到最小值 | 10 |
| WSD | warmup-stable-decay | 预热、长时间常数、最后短时间快速衰减 | 10 |
| 退火 | annealing / decay phase | 训练末段把 LR 降下来，loss 会明显再降一截 | 10、16 |
| 逐层分布式优化器 | layer-wise distributed optimizer | 按整个参数矩阵分给各卡做优化器更新；Muon 的正交化要看到完整矩阵，所以需要它 | 10、12 |

<a id="sa-10" name="sa-10"></a>
## A.10 数值精度

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 符号位、指数位、尾数位 | sign / exponent / mantissa | 浮点数的三部分；指数位定范围，尾数位定精度 | 11 |
| FP32 | FP32 | 8 位指数、23 位尾数；训练中主权重、优化器状态和累加都用它 | 11 |
| FP16 | FP16 | 5 位指数、10 位尾数，范围小，要配 loss scaling | 11 |
| BF16 | BF16 | 8 位指数、7 位尾数，范围同 FP32，精度低 | 11 |
| E4M3 | FP8 E4M3 | 4 位指数、3 位尾数，最大 448，前向用 | 11 |
| E5M2 | FP8 E5M2 | 5 位指数、2 位尾数，范围更大，反向梯度用 | 11 |
| 混合格式 | hybrid FP8 | 前向 E4M3、反向 E5M2，ZGCM 的 `FP8_FORMAT=hybrid` | 11 |
| 混合精度训练 | mixed precision training | 大部分计算用低精度，主权重、优化器状态、累加、归一化等用 FP32 | 11 |
| 主权重 | master weights | 优化器更新用的 FP32 权重副本 | 11 |
| loss 缩放 | loss scaling | 把 loss 乘一个大数防止 FP16 梯度下溢；BF16 一般不需要 | 11 |
| 缩放因子 | scaling factor | FP8 转换前给张量乘的系数，把数值搬进 FP8 的表示范围 | 11 |
| 绝对值最大值 | amax | 一个张量里绝对值最大的元素 | 11 |
| 延迟缩放 | delayed scaling | 用过去若干步的 amax 定这一步的缩放因子，ZGCM 历史长度 1024 | 11 |
| 当前缩放 | current scaling (`tensorwise`) | 每次转换前扫一遍当前张量求 amax 定缩放因子，更准但多一次读 | 11 |
| 细粒度缩放 | blockwise scaling | 每一小块各配一个缩放因子，离群值只影响所在小块，如 DeepSeek-V3 | 11 |
| 激活离群值 | activation outlier | 少数特别大的激活值，会把缩放因子压小，其余数精度变差 | 11 |
| TWEO | Transformers Without Extreme Outliers | 在 block 输出上加一个 loss 项压住激活离群值，ZGCM 报告说配合 delayed scaling 使用 | 11 |
| z-loss | z-loss | $Z$ 是 softmax 的分母（所有 logits 取指数后求和）；z-loss 惩罚 $\log Z$ 变大，防止输出 logits 整体漂移 | 11、14 |

<a id="sa-11" name="sa-11"></a>
## A.11 分布式并行

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 全归约 | all-reduce | 每张卡的张量求和，结果发回所有卡 | 12 |
| 归约散射 | reduce-scatter | 求和后每张卡只拿结果的一段 | 12 |
| 全收集 | all-gather | 每张卡出一段，拼成整个张量发给所有卡 | 12 |
| 全交换 | all-to-all | 每张卡把自己的数据切成份，分别发给每张卡 | 12 |
| 数据并行 | DP, data parallelism | 每张卡一份完整模型、处理不同数据，梯度 all-reduce | 12 |
| ZeRO-1/2/3 | ZeRO stage 1/2/3 | 让 DP 的各张卡不再各存一整份：stage 1 切优化器状态，stage 2 再切梯度，stage 3 连参数也切 | 12 |
| FSDP | fully sharded data parallel | PyTorch 版的 ZeRO-3 | 12 |
| 分布式优化器 | distributed optimizer | Megatron 版的 ZeRO-1，切优化器状态和主权重 | 12 |
| 张量并行 | TP, tensor parallelism | 把每个矩阵按行或列切到几张卡上，ZGCM 是 TP 2 | 12 |
| 序列并行 | SP, sequence parallelism | 在 TP 组内把归一化、dropout 等的激活按序列切开 | 12 |
| 上下文并行 | CP, context parallelism | 把一条序列切给几张卡，注意力时交换 K、V 或按头重排 | 12 |
| 环形注意力 | ring attention / p2p CP | CP 的一种，K、V 块沿环逐卡传递 | 12 |
| a2a 型 CP | all-to-all CP (Ulysses) | 用 all-to-all 把"按序列切"换成"按头切"；ZGCM 因 SWA 只能用这种 | 12 |
| 流水线并行 | PP, pipeline parallelism | 把层切成几段放在不同卡上，micro batch 依次流过 | 12 |
| 流水线气泡 | pipeline bubble | PP 开头和结尾部分卡在等待的空闲时间 | 12 |
| 微批大小 | MBS, micro batch size | 一张卡一次前向反向的序列条数，ZGCM 预训练是 2 | 12 |
| 梯度累积 | gradient accumulation | 多个 micro batch 的梯度累加后再更新一次；ZGCM $768 = 2\times48\times8$ | 12 |
| 激活值 | activations | 前向时存下来、反向要用的中间结果 | 12 |
| 激活重算 | recompute / activation checkpointing | 前向不存部分激活，反向时再算一遍，用计算换显存 | 12 |

<a id="sa-12" name="sa-12"></a>
## A.12 启动、检查点、监控

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| torchrun | torchrun | PyTorch 的多进程启动器，每张卡起一个进程并设好 rank 等环境变量 | 13 |
| 进程编号 | rank / local rank / world size | 全局第几个进程、本机第几个进程、总进程数 | 13 |
| env 文件 | env file | 只写参数取值的 bash 文件，由 launcher 读进来拼成命令行 | 13 |
| 启动脚本 | launcher | 读 env、做检查、拼参数、调 torchrun 的脚本，ZGCM 是 `launch_train.sh` | 13 |
| 进度文件 | `latest_checkpointed_iteration.txt` | 检查点目录里只有一行的文件，记最新的步数；加载时先读它 | 13 |
| torch_dist 格式 | torch_dist checkpoint | Megatron 的分布式检查点格式，按张量的全局形状存，换并行切法也能加载 | 13 |
| 续训 | resume | 从检查点接着训，权重、优化器状态、步数、consumed_samples 全部接上，和没停过一样 | 13 |
| 微调式加载 | finetune load (`--finetune`) | 只加载权重，优化器和步数从 0 开始；ZGCM mid-train 用它，和"做 SFT"无关 | 13 |
| 检查点间隔 | checkpoint interval | 多少步存一次；按 $\sqrt{2CM}$ 估，$C$ 是存一次的时间，$M$ 是平均故障间隔 | 13 |
| 垃圾回收 | gc, garbage collection | Python 定期停下来回收内存；各进程不同步时会拖慢所有人 | 13 |
| 显存碎片 | allocator fragmentation | 空闲显存被切成很多小块，大块申请变慢或失败 | 13 |
| 吞吐 | throughput | 每秒处理的 token 数或每卡 TFLOP/s | 1、12、13 |
| 慢节点 | straggler | 比别人慢的那张卡或那台机器，同步训练里所有人都等它 | 14 |
| loss 尖峰 | loss spike | loss 突然大幅上升，可能自己回落，也可能发散 | 14 |
| 发散 | divergence | loss 升上去不再回来，或出现 NaN | 14 |
| NaN | not a number | 数值溢出或 0/0 的结果，一出现就会传遍整个模型 | 14 |
| 回滚 | rollback | 从尖峰之前的检查点重启，常配合跳过那段数据或降 LR | 14 |
| 训练 loss 与验证 loss | train loss / validation loss | 在训练 batch 上和在固定的留出集上算的 loss | 14 |

<a id="sa-13" name="sa-13"></a>
## A.13 后训练：mid-train、SFT、RL

### Mid-train 与长上下文

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 扩长 | context extension | 让短序列训出的模型能处理更长的输入 | 16 |
| 分段扩长 | staged extension | 序列长度分几段逐步加长，ZGCM 实际是 32K→64K→128K→256K | 16 |
| 调大 base | RoPE base scaling (ABF) | 位置编号不动，只把 $\theta$ 调大让旋转变慢，再在新长度上接着训；ZGCM 的做法 | 16 |
| 位置插值 | PI, position interpolation | 把位置编号按比例压缩进训练过的范围 | 16 |
| NTK 感知缩放 | NTK-aware scaling | 按公式调大 RoPE 基数：转得最慢的几维被压到和位置插值一样，转得最快的几维几乎不变，中间平滑过渡 | 16 |
| YaRN | YaRN | 按频率分段处理 RoPE 并调整注意力温度的扩长方法，ZGCM 没用 | 16 |
| 重新预热 | re-warm | 新阶段从较小的值重新 warmup，学习率升回峰值，会让 loss 先上去一点 | 10、16 |
| 大海捞针 | NIAH, needle in a haystack | 在长文里藏一条信息，让模型把它找出来的长上下文测试 | 16 |
| RULER | RULER | 一组合成长上下文任务，比大海捞针难 | 16 |
| 文档内掩码 | intra-document mask | 打包时让注意力不跨文档 | 8、16 |

### SFT

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| loss 掩码 | loss mask | 标出哪些 token 算 loss，SFT 只给助手回复置 1 | 17 |
| 角色标记 | role tokens | 对话模板里标 system、user、assistant 的特殊 token | 3、17 |
| THD 格式 | THD layout | 把一个 pack 里所有 token 排成一维，张量形状是（总 token 数 T × 头数 H × 头维 D），配 cu_seqlens 用 | 17 |
| think 模式 | think / no-think | think 模式回答前先在 `<think>` 里写推理过程，no-think 模式直接回答 | 17 |
| 多轮对话 | multi-turn | 一条样本里有多次用户和助手交替 | 17 |
| 过拟合 | overfitting | 训练集 loss 继续降，没见过的数据上变差 | 17 |
| 工具调用 | tool call | 模型按约定格式输出函数名和参数，由环境执行后把结果喂回去 | 17、19 |

### RL

| 中文 | 英文 | 定义 | 章 |
|---|---|---|---|
| 策略 | policy | 正在训练的模型本身：给定前文，输出下一个 token 的分布 | 18 |
| 状态、动作 | state / action | 状态是当前前文，动作是生成的下一个 token | 18 |
| 奖励 | reward | 对一个完整回答打的分 | 18 |
| 可验证奖励 RL | RLVR, RL with verifiable rewards | 奖励由程序判定，如数学答案比对、代码跑测试 | 18 |
| 钻空子 | reward hacking | 模型找到拿高分但没真正做对的办法 | 18 |
| 采样生成 | rollout | 用当前策略对题目生成回答，交给判分器 | 18 |
| REINFORCE | REINFORCE | 按奖励加权提高所生成回答的对数概率的基础策略梯度方法 | 18 |
| 基线 | baseline | 从奖励里减掉的参考值，不改期望，只降方差 | 18 |
| 优势 | advantage | 奖励减去基线，表示这个回答比平均好多少 | 18 |
| 信用分配 | credit assignment | 一个回答的奖励该算到哪些 token 头上 | 18 |
| 价值网络 | critic / value network | PPO 里估计每个状态期望奖励的模型 | 18 |
| PPO | proximal policy optimization | 用重要性比值加裁剪限制每次更新幅度的策略梯度方法 | 18 |
| 重要性比值 | importance ratio | 新策略和采样时策略给同一 token 的概率之比 | 18 |
| 裁剪 | clip, $\varepsilon$ | 比值已经超出 $[1-\varepsilon, 1+\varepsilon]$、还想往同一方向推时不再给梯度，ZGCM 的 $\varepsilon=0.2$ | 18 |
| KL 惩罚 | KL penalty | 在奖励里减一项 KL 散度（衡量两个输出分布差多远），不让策略离参考模型太远；ZGCM 系数为 0 | 18 |
| 参考模型 | reference model | 冻结的初始策略，用来算 KL | 18 |
| GRPO | group relative policy optimization | 同一题采一组回答，用组内均值和标准差算优势，不要价值网络 | 18 |
| 组大小 | group size | 每题采几个回答，ZGCM 是 8 | 18 |
| 动态采样 | dynamic sampling | 丢掉全对或全错的组，因为它们的优势全是 0 | 18 |
| 长度惩罚 | length penalty / length shaping | 对答对但写得长的回答扣一点分 | 18 |
| 异步 RL | asynchronous RL | 生成和训练同时跑，生成用的权重会落后几步 | 18 |
| 离策略 | off-policy | 训练用的数据不是当前策略生成的 | 18 |
| 解耦 loss | decoupled loss | 区分生成数据时的"行为策略"和本次训练开始时的"近端策略"：前者与后者的比值纠正数据过旧，当前策略与近端策略的比值负责裁剪 | 18 |
| AReaL | AReaL | ZGCM 用的异步 RL 框架 | 18 |
| vLLM | vLLM | 常用的推理引擎，RL 里负责 rollout | 18 |
| 熵塌缩 | entropy collapse | 策略越来越确定，采样多样性消失，学不到新东西 | 18 |
| 过程奖励模型 | PRM, process reward model | 给推理的每一步打分，而不是只看最终答案 | 18 |
| DPO | direct preference optimization | 不用 RL，直接用"好回答对坏回答"的偏好对训练 | 18 |

---

上一章：[第 20 章　总复习与讲出来](20_总复习与讲出来.md)　下一章：[附录 B　数字速查](附录B_数字速查.md)

