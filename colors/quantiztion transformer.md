推理优化通常有下面三个目的
  ○ 减少内存使用（model size）
  ○ 降低模型计算复杂度（FLOPs）
  ○ 减少推理时间 （runtime）
对于显存占用比较大的transformer类大模型，可以从下面几类方法入手来加快推理速度：
  ○ Parallelism（模型并行，数据并行，单机多卡，多机多卡）
  ○ Memory offloadding（用CPU暂存数据）
  ○ Smart batching strategy （更好的batch策略）
  ○ Model compression （量化，蒸馏，剪枝，稀疏）
  ○ Architecure optimization (特定pattern的模型结构优化)
Model compression
Quantization
量化transformer的主要难点：由于activation的动态范围很大，简单的PTQ 8bit量化会带来比较大的精度损失。因此保持activation为分fp32（W8A32），通常能带来更好的结果。（https://arxiv.org/abs/2109.12948）
随着模型尺寸的增大，activaiton的动态范围越大，在FFN的输出分布中出现越来越多的离群点，导致transformer在8bit 的量化时候效果很差。（https://arxiv.org/abs/2208.07339）
PTQ
Mixed-precsison Quantization
LLM.int8() （https://arxiv.org/abs/2208.07339）：矩阵X和W的每行每列分别量化，离群点保持FP16计算

其他采用混合精度量化的方案：
  ○ Understanding and Overcoming the Challenges of Efficient Transformer Quantization
  ○ TODO
Quantization at fine-grained granularity
采用更细粒度的量化分组，缺点是实现起来更为复杂，尤其是在dequant的时候，需要用到不同组的量化参数。

Q-BERT 对Mutil-head attention中的每组W采用不同的量化参数（per-group）
Per-embedding group 则是在d的每个维度寻找量化参数， 或者将d维度分组，每组用相同的计算参数。
参考：
  ○ Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT
  ○ ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers
Second order imformation for quantization
用Hessian矩阵提供的二阶信息辅助量化在很多论文中都有类似操作。其出发点每一层的Hessian矩阵中特征值较大的点对量化结果的影响更大，Q-BERT中就采用了Hessian矩阵来进行混合精度的量化。
GPTQ把weight的量化看作一个优化问题，weight的每行单独量化，迭代的寻找最优的量化w，也用到了Hessian来更新weight_q

其他用到Hessian matrices的方法：
  ○ TODO
Outlier smoothing
对于activation中的离群点，可以通过对activation整体除以系数s，对weight乘以s，使得activation的量化误差一定程度上减少。

放缩系数s的计算公式，论文中alpha=0.5


参考：
  ○ SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models
其他PTQ
paper：
  ○ Patch Similarity Aware Data-Free Quantization for Vision Transformers ECCV2022
  ○ FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer IJCAI2022
  ○ GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers ICLR2023
github：
● OpenNMT/CTranslate2
● PanQiWei/AutoGPTQ
● IST-DASLab/gptq
● Qualcomm-AI-research/transformer-quantization
● huggingface/optimum
● Sharpiless/Point-Transformers-with-Quantization
● PaddlePaddle/PaddleSlim
● hahnyuan/PTQ4ViT
● zkkli/PSAQ-ViT


QAT
● 传统QAT方案
● 用fp32模型做distillation得到int8模型的方案
Distillation
知识蒸馏：在给定的数据集下，用一个student model 来近似teacher model 的output，通过训练student model最小化distillation loss，让二者的输出尽量接近。

通常而言，选取模型softmax层的输出分布作为依据，蒸馏loss可以说KL散度或cross entropy。

参考：DistilBERT (https://arxiv.org/abs/1910.01108)，在特定数据集下蒸馏后得到的小模型减少了40%的参数，达到97%的精度，比原始模型快71%。
Pruning
● Magnitude pruning
Sparsity (TODO)	
● N:M Sparsity via pruning
● sparsified transformer 
● Mixture of experts
Architecture Optimization
  ○ Efficient Transformers: A Survey



