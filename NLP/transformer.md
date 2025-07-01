- [transformer相关知识点总结](#transformer相关知识点总结)
  - [1.Attention](#1attention)
  - [2.位置编码](#2位置编码)
  - [3.为什么BERT选择mask掉15%这个比例的词，可以是其他的比例吗？](#3为什么bert选择mask掉15这个比例的词可以是其他的比例吗)
  - [4.使用BERT预训练模型为什么最多只能输入512个词，最多只能两个句子合成一句？](#4使用bert预训练模型为什么最多只能输入512个词最多只能两个句子合成一句)
  - [5.为什么BERT在第一句前会加一个[CLS]标志?](#5为什么bert在第一句前会加一个cls标志)
  - [5.bert的特点](#5bert的特点)
  - [6.bert缺点](#6bert缺点)
  - [7.XLNet](#7xlnet)
    - [自回归语言模型](#自回归语言模型)
    - [自编码语言模型](#自编码语言模型)
  - [8.ALBERT](#8albert)
  - [9.transformer-xl](#9transformer-xl)
- [面试问题](#面试问题)

# transformer相关知识点总结

## 1.Attention 
- [一文看懂 Attention（本质原理+3大优点+5大类型）](https://zhuanlan.zhihu.com/p/91839581)
- [拆 Transformer 系列二：Multi- Head Attention 机制详解](https://zhuanlan.zhihu.com/p/109983672)
- [Self-Attention 的时间复杂度是怎么计算的？](https://zhuanlan.zhihu.com/p/132554155) n^2d, n句子长度，d维度
- [Transformer的点积模型做缩放的原因是什么？](https://www.zhihu.com/question/339723385):向量的点积结果会很大，将softmax函数push到梯度很小的区域，scaled会缓解这种现象
- [不考虑多头的原因，self-attention中词向量不乘QKV参数矩阵，会有什么问题？]: 1. q*k自身的值会很大，在softmax后，该词本身所占的权重很大， 2. 多头机制，类似于cnn的多核卷积
- multi-head:多头可以使参数矩阵形成多个子空间，矩阵整体的size不变，只是改变了每个head对应的维度大小，这样做使矩阵对多方面信息进行学习，但是计算量和单个head差不多
- .Transformer为什么Q和K使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘:请求和键值初始为不同的权重是为了解决可能输入句长与输出句长不一致的问题。并且假如QK维度一致，如果不用Q，直接拿K和K点乘的话，你会发现attention score 矩阵是一个对称矩阵。因为是同样一个矩阵，都投影到了同样一个空间，所以泛化能力很差
- 10*64的输入，mutil-head attention，大概多少参数量？

## 2.位置编码

- [让研究人员绞尽脑汁的Transformer位置编码](https://zhuanlan.zhihu.com/p/352898810)

## 3.为什么BERT选择mask掉15%这个比例的词，可以是其他的比例吗？

- 1.BERT采用的Masked LM，会选取语料中所有词的15%进行随机mask，论文中表示是受到完形填空任务的启发，但其实与CBOW也有异曲同工之妙。
- 2.从CBOW的角度，这里 1/0.15= 7 有一个比较好的解释是：在一个大小为 [公式] 的窗口中随机选一个词，类似CBOW中滑动窗口的中心词


## 4.使用BERT预训练模型为什么最多只能输入512个词，最多只能两个句子合成一句？
- 首先说明embedding包含3个部分，token,position,segment
- bert config的配置 
- 

## 5.为什么BERT在第一句前会加一个[CLS]标志?

BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。

为什么选它呢，因为与文本中已有的其它词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。

## 5.bert的特点

bidirection encoder representation of transformer

- 双向：能同时利用当前单词的上下文信息来做特征提取， 对比rnn, cnn
- 动态表征: ：利用单词的上下文信息来做特征提取，根据上下文信息的不同动态调整词向量
- 并行运算的能力
- 易于迁移学习:使用预训练好的BERT，只需加载预训练好的模型作为自己当前任务的词嵌入层，后续针对特定任务构建后续模型结构即可，不需对代码做大量修改或优化


## 6.bert缺点
- 第一个预训练阶段因为采取引入[Mask]标记来Mask掉部分单词的训练模式，而Fine-tuning阶段是看不到这种被强行加入的Mask标记的，所以两个阶段存在使用模式不一致的情形，这可能会带来一定的性能损失
- Bert在第一个预训练阶段，假设句子中多个单词被Mask掉，这些被Mask掉的单词之间没有任何关系，是条件独立的，而有时候这些单词之间是有关系的，XLNet则考虑了这种关系
- 句子长度限制，必须分段编码


## 7.XLNet
### 自回归语言模型
上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词


### 自编码语言模型
Bert通过在输入X中随机Mask掉一部分单词，然后预训练过程的主要任务之一是根据上下文单词来预测这些被Mask掉的单词

bert这种自编码语言模型的缺点:1.预训练[mask], finetune没有，有差，性能损失; 2.bert中是随机mask,但是这些词有关系，xlnet考虑了这种关系。

[XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

## 8.ALBERT
- Factorized Embedding Parameterization
- Cross-layer Parameter Sharing
- Sentence Order Prediction

[如何看待瘦身成功版BERT——ALBERT？](https://www.zhihu.com/question/347898375/answer/863537122)


## 9.transformer-xl
transformer的缺点: 对长句子的编码能力
- 长句子切割必然会造成语义的残破，不利于模型的训练
- segment的切割没有考虑语义，也就是模型在训练当前segment时拿不到前面时刻segment的信息，造成了语义的分隔

**如何解决**: Recurrence机制，在计算该时刻的状态时，引入前一时刻的状态作为输入

Trm-XL为了解决长序列的问题，对上一个segment做了缓存，可供当前segment使用，但是也带来了位置关系问题，为了解决位置问题，又打了个补丁，引入了相对位置编码

- [Transformer-XL介绍](https://zhuanlan.zhihu.com/p/84159401)
- [NLP（9）：TransformerXL：因为XL，所以更牛](https://zhuanlan.zhihu.com/p/78351660)





# 面试问题
- [超细节的BERT/Transformer知识点](https://zhuanlan.zhihu.com/p/132554155)
- [在BERT应用中，如何解决长文本问题？](https://www.zhihu.com/question/327450789)
- [NLP 中的Mask全解](https://zhuanlan.zhihu.com/p/139595546)
- [transformer面试题的简单回答](https://zhuanlan.zhihu.com/p/363466672)