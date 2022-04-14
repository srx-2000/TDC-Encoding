### 二维卷积在生成式摘要生成中的使用

该项目为我本科毕设

毕设要求如下：

1. 对比分析并复现出一到两个摘要生成模型
2. 构建一个改进的生成式摘要模型
3. 构建一个简单易用的摘要自动生成系统

可以看出主要的任务难度分布在了前两个任务上，所以最近这三个月的时间，我可以说是从0开始学习关于该领域的相关知识，从深度学习最基础的概念，到pytorch框架的使用，再到论文的阅读与复现。终于在一个礼拜前我构思并初步实现了我自己理解中的一个模型，架构图如下图所示：![](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/1.png)

​	主要参考了以下两篇论文中的思路：[Text Summarization with Pretrained Encoders](https://arxiv.org/pdf/1908.08345.pdf)，[Global Encoding for Abstractive Summarization](https://arxiv.org/pdf/1805.03989.pdf)，这两篇论文也都在github上有相应的代码库：[paper1](https://github.com/nlpyang/PreSumm)，[paper2](https://github.com/lancopku/Global-Encoding)。我在二者中获取了一些灵感并尝试将二者的优点有机的结合了以下，同时我也尝试提出一个相对比较冷门的想法：将二维卷积使用在nlp任务中。

​	众所周知，nlp中使用卷积更多的是使用一维卷积对单句进行卷积计算，而更高维度的维卷积则很少使用在nlp处理中，而由于本次任务是：将较长的文本输入网络并自回归生成，所以这里其实是根据我个人的第一直觉从而产生了使用将句子进行分割，并将分割后的segment堆叠成为一个二维平面，并以此平面为基准进行二维卷积，效果如下图所示【整体架构中的convolution unit模块】：

![](https://raw.githubusercontent.com/srx-2000/Two-dimensional-convolution-for-Abstractive-summarization/master/2.png)

​	但由于是第一次写深度学习的代码，所以可能代码可读性比较差，同时该模型还没有进行模型调优与具体参数优化，所以效果非常差，后续可能会持续更新调整参数，架构等。希望本项目能作为大家在摘要生成领域的一块板砖，引出大家珠玉。

​	如果大家对我处理好的数据集有需求的话，请使用百度云盘进行下载：https://pan.baidu.com/s/1fiOeRl4Ty7HE1CvoW3QtXQ【密码：1111】