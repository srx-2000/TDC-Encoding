"""
@ModuleName: BertModel
@Description: 
@Author: Beier
@Time: 2022/4/4 11:10
"""
from torch import nn
from transformers import BertConfig, BertModel


class BertModels(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=1):
        super(BertModels, self).__init__()
        print("bert模型加载中.....")
        self.bert = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=r"../../temp")
        self.bert_config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=r"../../temp")
        self.con1 = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride)
        print(self.bert)
        print(self.bert_config)

    def forward(self, inputs):
        return self.con(inputs)


if __name__ == '__main__':
    bm = BertModels(100, 1)
