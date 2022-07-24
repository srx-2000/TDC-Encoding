### 二维卷积在生成式摘要生成中的使用

该项目为我本科毕设的模型模块

下面是各个部分的库连接：

[系统爬虫模块]()

[系统后端模块]()

[系统前端模块]()

[毕设所有材料【包含论文、答辩ppt、所有图】]()

毕设要求如下：

1. 对比分析并复现出一到两个摘要生成模型
2. 在复现模型的基础上改进的一个生成式摘要模型
3. 构建一个简单易用的摘要自动生成系统

可以看出主要的任务难度分布在了前两个任务上，所以最近这三个月的时间，我可以说是从0开始学习关于该领域的相关知识，从深度学习最基础的概念，到pytorch框架的使用，再到论文的阅读与复现。终于在一个礼拜前我构思并初步实现了我自己理解中的一个模型，架构图如下图所示：![模型整体架构](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/1.png)

主要参考了以下两篇论文中的思路：[Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)，[Global Encoding for Abstractive Summarization](https://arxiv.org/pdf/1805.03989.pdf)，这两篇论文也都在github上有相应的代码库：[paper1](https://github.com/nlpyang/PreSumm)，[paper2](https://github.com/lancopku/Global-Encoding)。我在二者中获取了一些灵感并尝试将二者的优点有机的结合了以下，同时我也尝试提出一个相对比较冷门的想法：将二维卷积使用在nlp任务中。

#### 二维卷积层

众所周知，nlp中使用卷积更多的是使用一维卷积对单句进行卷积计算，而更高维度的维卷积则很少使用在nlp处理中，而由于本次任务是：将较长的文本输入网络并自回归生成，所以这里其实是根据我个人的第一直觉从而产生了使用将句子进行分割，并将分割后的segment堆叠成为一个二维平面，并以此平面为基准进行二维卷积，效果如下图所示【整体架构中的二维卷积层模块】：

![二维卷积层](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/2.png)

同时为了更好地将nlp中的句向量与二维卷积层进行适配，这里提出了另外三个适配组件：TDC Embedding，片段映射，融合自注意机制。

#### TDC embedding

所谓TDC Embedding即是，将经典循环神经网络中的embedding一分为二，变成：句嵌入（sentence embedding）、片段嵌入（segments embedding）。其中句嵌入将会输入到双向LSTM中学习句粒度的特征信息。而片段嵌入则如下图中所示，将长句进行切分堆叠最终形成一个文本矩阵，进而转化为词向量矩阵，而该矩阵将会作为二维卷积层的输入，进一步学习片段粒度的语义特征。

![TDC Embedding1](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/3.png)

为了便于大家理解，这里给出处理后与二维卷积对接时的block图

![TDC Embedding2](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/4.png)

上图显示的则是将上述文本矩阵进行词嵌入后的结果。如图所示，单个句子的文本矩阵会变为一个三维矩阵。从而得以进入二维卷积层进行计算。

#### 片段映射

片段映射在模型中主要担当了将双向LSTM的输出与卷积神经网络的输出进行映射的任务。其核心思想十分类似于Transformer中的自注意力机制。其中融合了完整语义的双向LSTM输出，作为映射中的上下文信息，而更注重局部语义的卷积神经网络输出，则作为查询者。其公式如下所示。公式为：
$$
VMS(Q,K,V)=\frac{QK^T}{d_k}V
$$
​      其中Q代表作为查询者的卷积神经网络输出。而K和V则是代表了作为上下文信息的双向LSTM输出。最后为了便于后续梯度计算，这里使用dk进行归一化。

#### 融合自注意力机制

通过片段映射后的输出虽然已经融合了卷积神经网络与循环神经网络中的特征，但其维度本身并未有太大变化，仍然是一个四维的向量。为了便于后续解码器的计算，同时也为了有利于建立长依赖关系，使每个位置的词向量都有全局的语义信息，这里引入融合自注意力机制。

![融合自注意力机制](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/5.png)

融合自注意力机制实质是在自注意力机制的基础上增加了一个特征融合的计算，使得从二维卷积中得到的输出能与片段映射后的输出进行特征融合，并最终将融合后的结果展开至三维向量，以便后续解码器进行运算

#### 实验效果

这里在最终的实验效果比对上，主要与其余三种模型做了对比，分别是：[Global Encoding](https://arxiv.org/pdf/1805.03989.pdf)、[BERTSUM-EXT-ABS](https://arxiv.org/pdf/1908.08345.pdf)以及[SumGAN](https://arxiv.org/pdf/1711.09357.pdf)。

<table>
    <tr>
        <td style="text-align: center;"><b>模型</b></td> 
        <td style="text-align: center;"><b>ROUGE-1</b></td> 
        <td style="text-align: center;"><b>ROUGE-2</b></td>
        <td style="text-align: center;"><b>ROUGE-3</b></td>
   </tr>
    <tr>
        <td style="text-align: center;">SumGAN</td>
        <td style="text-align: center;">39.92</td>  
        <td style="text-align: center;">17.65</td>  
        <td style="text-align: center;">36.71</td>  
    </tr>
    <tr>
        <td style="text-align: center;">BERTSUM-EXT-ABS</td>
        <td style="text-align: center;">42.13</td>  
        <td style="text-align: center;">19.6</td>  
        <td style="text-align: center;">39.18</td>  
    </tr>
    <tr>
        <td colspan="4" style="text-align: center;"><b>CNN/DailyMail英文数据集</b></td>    
    </tr>
    <tr>
        <td style="text-align: center;">Global Encoding</td>
        <td style="text-align: center;">39.4</td>  
        <td style="text-align: center;">26.9</td>  
        <td style="text-align: center;">36.5</td>  
    </tr>
    <tr>
        <td style="text-align: center;">TDC Encoding</td>
        <td style="text-align: center;">42.08</td>  
        <td style="text-align: center;">27.75</td>  
        <td style="text-align: center;">38.84</td>  
    </tr>
    <tr>
        <td colspan="4" style="text-align: center;"><b>LCSTS中文数据集</b></td>    
    </tr>
</table>

#### 总结

以上便是TDC Encoding模型的全部架构以及创新点，模型整体架构参考了Global Encoding的模型架构，同时还吸收了BERTSUM-EXT-ABS模型的片段切分思想【即文档输入，切分segments，进行embedding】，从最终效果的角度上来讲相较于作为主要参考的Global Encoding模型，评分上有了一定的进步。可以见得二维卷积模型在nlp领域也应具有一定的效果。

#### 未来改进方向

最后在这里给出我个人对该模型的未来展望：

1. 由于使用TED embedding的缘故，导致模型输入的矩阵多为稀疏矩阵，有待改进，将其压缩为稠密矩阵。
2. 后经毕设评阅老师提醒，发现在做TDC embedding时可以，将句子优先翻转再进行堆叠。这样处理后，原本语义中相邻的向量就不会因为矩阵堆叠而减少相关性。
3. 如果有可能想尝试使用Gan网络对模型生成的摘要进行全局优化，其实该想法在设计架构时就有产生，但后续由于毕设完成周期太短而未能实现。之所以有这个想法，是因为卷积更注重的是向量间的局部特征，而Gan网络则是可以对生成的语句进行宏观评价。同时模型[SumGAN](https://arxiv.org/pdf/1711.09357.pdf)也从一定角度上验证了Gan网络是可以应用在nlp领域内的。
4. 最后便是模型调优，由于毕设给的时间太少导致虽然将模型整体搭了出来，但并没有时间进行调优，基本的参数都沿用了Global Encoding模型的参数。如有朋友对该模型调优感兴趣的话，建议可以优先尝试参数调优。

​	最后由于是第一次写深度学习的代码，所以可能代码可读性比较差，同时该模型还没有进行模型调优与具体参数优化，所以效果可能并没有达到最优解。希望本项目能作为大家在摘要生成领域的一块板砖，引出大家珠玉。

​	如果大家对我处理好的数据集、训练好的模型有需求的话，请使用百度云盘进行下载：https://pan.baidu.com/s/1lc9_9cQmRo4Yv7G_f3PCUw【密码：1234】