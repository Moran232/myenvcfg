NLP资料整理
词表示法
视觉任务中用图像像素表示的矩阵作为输入，语音任务中用音频频谱图表示的矩阵作为输入。而语言高度抽象，单词、句子、段落和文本都需要用合适的编码方法表示为向量或矩阵，作为后续模型输入
onehot编码
 
•	特点：
o	词向量长度等于词典大小，表征能力有限，新词加入后需要重新生成编码
o	词与词之间的关系在编码后没有体现，仅仅是把不同词作了区分
分布式表示distributed word representation 
分布假设：具有相似语义的两个词/句子/文本，在编码后的分布应该相似
Brow cluster
Latent semantic analysis(LSA)
将包含全局信息的文档-术语矩阵，进行矩阵奇异值分解，降低矩阵维度。
参考：
•	https://arxiv.org/abs/2108.06197
•	https://www.jiqizhixin.com/articles/2019-06-12-8
Word2Vec
出自Google 2013年的论文《Efficient Estimation of Word Representations in Vector Space》，本质上是通过训练的方法把长度为V（Vocabulary）的one-hot稀疏词向量映射到长度为N(N<<V)的向量空间中去（word embedding），N维空间中距离相近的向量表达的词义相近。具体用到的方法是连续词袋模型CBOW和Skip-gram，为了加速训练论文中还用到了Hierarchical softmax，negative sampling, Huffman Tree等
•	Continuous bag-of-words
利用上下文来预测中间词：用上下相邻的C个单词来预测中间单词，经过输入层矩阵W和输出层矩阵W’得到输出y，y经过softmax后与要预测的中间词的onehot编码做差得到损失函数，训练后得到的W就是每个单词的词向量。
 
参考：
o	Efficient Estimation of Word Representations in Vector Space
o	https://ask.hellobi.com/blog/wangdawei/36569
•	Skip-gram
与CBOW相反，用中间词预测上下文
 
参考：
•	http://xtf615.com/2018/10/05/word2vec/
•	https://paddlepedia.readthedocs.io/en/latest/tutorials/sequence_model/word_representation/word2vec.html
word2vec的出现，极大的促进了NLP的发展，尤其是促进了深度学习在NLP中的应用（不过有意思的是，word2vec算法本身其实并不是一个深度模型，它只有两层全连接），利用预训练好的词向量来初始化网络结构的第一层几乎已经成了标配，尤其是在只有少量监督数据的情况下，如果不拿预训练的embedding初始化第一层，几乎可以被认为是在蛮干。 虽然咿咿呀呀囫囵吞枣似的刚开始能够说得三两个词，然而这是“NLP的一小步，人类AI的一大步”。
在此之后，一大批word embedding方法大量涌现，比较知名的有GloVe和fastText等等，它们各自侧重不同的角度，并且从不同的方向都得到了还不错的embedding表征。
Golve
出自Stanford NLP Group的Jeffrey Pennington, Richard Socher, Christopher D. Manning在2014年的EMNLP上发表的一篇论文：GloVe: Global Vectors for Word Representation，综合利用全局特征的矩阵分解方法，和利用局部上下文的方法。
参考：
•	GloVe: Global Vectors for Word Representation
•	https://www.fanyeong.com/2018/02/19/glove-in-detail/
Fasttext
fastText是一种Facebook AI Research在16年开源的一个文本分类器，与CBOW结构相似，不同的是不再预测中间词而是预测一个标签。
•	Bag of Tricks for Efficient Text Classification
•	https://blog.csdn.net/GFDGFHSDS/article/details/105343100
ElMo
NAACL18 Best Paper，不同于以往的一个词对应一个向量是固定的，在ELMo世界里，预训练好的模型不再只是向量对应关系，而是一个训练好的模型。使用时，将一句话或一段话输入模型，模型会根据上线文来推断每个词对应的词向量。
参考
•	Deep contextualized word representations
•	https://zhuanlan.zhihu.com/p/51679783
Transformer之前的语言模型
传统的语言建模的研究出现在上个世纪，2003年Benjo首次将神经网络用在了语言建模中，2013年出现的Word2Vec方法将NLP
 
语言模型 
 
 
 
N-gram N元文法模型
 
N-gram总体存在三个问题：
第一，很多情况下公式的计算会遇到特别多零值，尤其是在n取值比较大的情况下，这种数据稀疏导致的计算为0的现象变得特别严重。所以统计语言模型中一个很重要的方向便是设计各种平滑方法来处理这种情况。
第二，基于统计的语言模型无法把n取得很大，一般来说在3-gram比较常见，再大的话，计算复杂度会指数上升。这个问题的存在导致统计语言模型无法建模语言中上下文较长的依赖关系。
第三，统计语言模型无法表征词语之间的相似性。
NNLM
2003年Bengio在他的经典论文《A Neural Probabilistic Language Model》中，首次将深度学习的思想融入到语言模型中，通过神经网络来估计n-gram中的条件概率。
NNLM(Neural Network Language Model)的最主要贡献是非常有创见性的将模型的第一层特征映射矩阵当做词的分布式表示，从而可以将一个词表征为一个向量形式，这直接启发了后来的word2vec的工作。
 
NNLM虽然将N-Gram的阶n提高到了5，相比原来的统计语言模型是一个很大的进步，但是为了获取更好的长程依赖关系，5显然是不够的。再者，因为NNLM只对词的左侧文本进行建模，所以得到的词向量并不是语境的充分表征。
还有一个问题就更严重了，NNLM的训练依然还是太慢，在论文中，Bengio说他们用了40块CPU，在含有1400万个词，只保留词频相对较高的词之后词典大小为17964个词，只训练了5个epoch，但是耗时超过3周。按这么来算，如果只用一块CPU，可能需要2年多，这还是在仅有1400万个词的语料上。如此耗时的训练过程，显然严重限制了NNLM的应用。
RNN
让一个词的计算包含之前所有词的信息，输入序列和输出序列必须是等长的。应用如视频每一帧的分类任务。
  
N vs. 1 的RNN
有时输入为N输出长度是1，需要一个从N到1到映射，此时只需要保留最有一个输出即可，通常用在序列分类任务中，如判断一段文本的类别，判别一段句子到情感倾向。
 
1 vs. N的RNN
从一个输入生成一段文本的情况，只用最开始输入x，或者每次输入的x都保持不变，应用如从图像生成文字，自动创作歌曲等
  
LSTM
为了解决经典RNN中存在的长期依赖问题（经过网络中多个节点的计算后，之前时间点的特征已经被覆盖掉），LSTM引入了输入门、遗忘门、输出门和记忆细胞，使得网络能够记忆之前较远时间处的信息。
 
参考：
•	https://zhuanlan.zhihu.com/p/42717426
•	http://colah.github.io/posts/2015-08-Understanding-LSTMs/
GRU
和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。
参考：
•	https://zhuanlan.zhihu.com/p/32481747
•	Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation
N vs. M的RNN（Seq2Seq）
输入长度为N，输出长度为M，也就是任意长度序列的映射的RNN，也可以叫Encoder-Decoder模型或者Seq2Seq。Seq2Seq被提出于2014年，最早由两篇文章独立地阐述了它主要思想，分别是Google Brain团队的《Sequence to Sequence Learning with Neural Networks》和Yoshua Bengio团队的《Learning Phrase Representation using RNN Encoder-Decoder for Statistical Machine Translation》。
Encoder将输入序列编码成一个上下文向量c，最简单的就是直接使用最后一个隐状态，拿到c后用另一个RNN也就是Decoder对c做解码，得到输出序列
 
还有一种做法是把c当作decoder每一步的输出
 
由于可以完成任意长度的序列的映射，被应用在很多方面，如：机器翻译；文本摘要，输入是一段文本序列，输出是这段文本序列的摘要序列；阅读理解，将输入的文章和问题分别编码，再对其进行解码得到问题的答案；语音识别，输入是语音信号序列，输出是文字序列。
参考：
•	Cho et al., 2014 . Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
•	Sutskever et al., 2014. Sequence to Sequence Learning with Neural Networks.
•	Bahdanau et al., 2014. Neural Machine Translation by Jointly Learning to Align and Translate.
•	Jean et. al., 2014. On Using Very Large Target Vocabulary for Neural Machine Translation.
•	Vinyals et. al., 2015. A Neural Conversational Model. Computer Science.
从2013年Word2Vec出现以后，基于RNN的LanguageModel得到长足发展，在机器翻译、文本生成、阅读理解、对话系统等多个方面取得广泛的应用。当然，除了RNN外也尝试过其他深度学习(CNN，强化学习，Memory-augmented深度学习等)方法来对语言序列进行建模，但总体来看Encoder-Decoder方法取得的效果最好。
但它依然有一个问题：对于任何一个输出词，所有的输入词的影响是一样的，也就是说输出时候没有聚集到某些特定的输入词语，这在很多场景是不合理的。例如翻译“机器学习”到“machine learning”，显然在输出machine的时候需要让“机器”占有大的比重，在输出“learning”的时候“学习”应起主导作用。这就是模型没有注意力机制。
Attention机制
前面提到Encoder-Decoder结构中Encoder的输出的是固定长度的向量C，我们很难寄希望于将输入的序列中的全部信息保存在这个定长的向量中，随着所需翻译句子的长度的增加，这种结构的效果会显著下降。2015 年 Bengio团队在论文《Neural Machine Translation by Jointly Learning to Align and Translate》提出了Attention机制。
Attention机制，又称为注意力机制，顾名思义，是一种能让模型对重要信息重点关注并充分学习吸收的技术，通俗的讲就是“关注重点”。其实从全联接网络到CNN以及CNN中的Pooling操作，都可以认为是Attention机制在空间中的表现。
Attention机制在Encoder-decoder中的应用，其实就是要在用c计算每次输出y时候，c不再是固定的序列，而是与输入有关。
   
那么这些aij是怎么得来的？答案是通过学习得到，aij与Decoder的第i-1阶段的隐状态、Encoder第j个阶段的隐状态有关。
  
这张图表达的比较清楚
 
 
 
参考：
•	https://www.cnblogs.com/gczr/p/14693829.html
•	https://zhuanlan.zhihu.com/p/28054589
•	https://zhuanlan.zhihu.com/p/42724582
引入Attention机制后的seq2seq模型取得了更好的效果，但是基于RNN结构的计算模式难以并行，制约了其进一步发展，Facebook甚至提出了基于卷积的seq2seq模型《Convolutional Sequence to Sequence Learning》(ICML2017，8 May 2017)，在几个翻译任务上都比RNN类的模型表现更好且效率更高。正当大家争论该用RNN还是CNN的时候，google再次站了出来《Attention is all you need》(12 Jun 2017)。
Transformer结构解析
尽管lstm gru一定程度上解决了长期依赖的问题，但是对特别长的序列依然无能为力，而且基于RNN的结构并行度很差，Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。作者的实验是通过搭建编码器和解码器各6层，总共12层的Encoder-Decoder，并在机器翻译中取得了BLEU值得新高。
原文中why self-attention提到为什么要用self-attention的三点，一是计算复杂度相比rnn和conv更低，二是计算的可并行度，三是path length between long-range dependencies。
RNN模型中，输入序列表征任意两个单词之间的信息计算复杂度是O(n)，而transformer中是O(1)
整体结构
encoder
单encoder结构
 
•	embedding + positon encoding
 
•	self-attention
 
•	multi-head self attention
 
 
feedforward network
 
 
•	layernorm and redensial
 
decoder
 
linear output
 
参考：
📎transformer_v1.pdf
http://jalammar.github.io/illustrated-transformer/
https://zhuanlan.zhihu.com/p/338817680
Transformer在CV中的应用
DETR
End-to-End Object Detection with Transformers（ECCV，2020）
大大简化了目标检测任务的实现过程，不再需要anchor、proposal、window center等先验知识，也不再需要nms等后处理操作，端到端的实现了目标检测。最大的两个贡献：用可学习的object query取代了proposal/anchor，用二分图匹配取代了nms。
 
 
1.	CNN 获得feature
2.	送入encoder获得全局信息
3.	decoder根据object query和encoder的输出生成检测框，最后生成的框数与object query相对应
4.	计算loss(分类loss+候选框loss)时只选择最接近ground truth的两个box计算loss，其他为背景类
使用encoder获得全局信息，使得模型能够将一个object整体识别在一个bounding box中，从而避免了冗余的anchors情况出现。取得了与Fast R-CNN comparable的结果，在大物体上预测结果好，小物体上预测结果差，也为后续工作留下改进空间。此外，DETR的训练和推理都比Faster R-CNN要慢。
20个训练好的object query在预测时候所关注的候选框情况
 
Deformable DETR：可变形的Transformer
Deformable DETR: Deformable Transformers for End-to-End Object Detection (ICLR2021)
 
ViT
An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale（Oct. 2020, ICLR2021）
ViT原论文中最核心的结论是，直接在图像的patches上训练，在大的数据集上预训练迁移到小数据集上，ViT的表现就会超过CNN，突破transformer缺少归纳偏置的限制，可以在下游任务中获得较好的迁移效果。但是当训练数据集不够大的时候，ViT的表现通常比同等大小的ResNets要差一些，因为Transformer和CNN相比缺少归纳偏置（inductive bias），即一种先验知识，提前做好的假设。
 
总而言之，大的预训练数据集加上大模型，是ViT取得SOTA性能的关键因素。
参考：
https://zhuanlan.zhihu.com/p/445122996
https://zhuanlan.zhihu.com/p/581812634
Transformer Quantization
