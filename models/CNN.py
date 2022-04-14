"""
@ModuleName: CNN
@Description: 
@Author: Beier
@Time: 2022/4/4 11:12
"""

from torch import nn
import torch
from my_model.util.dataLoader import DataLoader, DataSetMultiFile, Data_Prefetcher, BatchFix


class CNN(nn.Module):
    def __init__(self, stack_channel, output_channel, stride=1):
        super(CNN, self).__init__()
        self.stack_channel = stack_channel
        self.output_channel = output_channel
        # self.kernel_size = kernel_size
        self.stride = stride
        self.sw1 = nn.Sequential(nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=1, padding=0),
                                 nn.BatchNorm3d(self.output_channel), nn.ReLU())

        self.sw3 = nn.Sequential(nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=1, padding=0),
                                 nn.ReLU(), nn.BatchNorm3d(self.output_channel),
                                 nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=3, padding=1),
                                 nn.ReLU(), nn.BatchNorm3d(self.output_channel))

        self.sw33 = nn.Sequential(nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=1, padding=0),
                                  nn.ReLU(), nn.BatchNorm3d(self.output_channel),
                                  nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=3, padding=1),
                                  nn.ReLU(), nn.BatchNorm3d(self.output_channel),
                                  nn.Conv3d(self.stack_channel, self.output_channel, kernel_size=3, padding=1),
                                  nn.ReLU(), nn.BatchNorm3d(self.output_channel))
        self.filter_linear = nn.Linear(3 * self.output_channel, self.output_channel)
        # self.con1 = nn.Conv2d(self.stack_channel, self.output_channel, kernel_size=self.kernel_size, stride=self.stride)
        # self.relu = nn.ReLU()
        # self.max_pool = nn.MaxPool2d(self.kernel_size)
        # self.transformer = nn.Transformer()

    def forward(self, sentence_feature):
        conv1 = self.sw1(sentence_feature)
        conv2 = self.sw1(sentence_feature)
        conv3 = self.sw1(sentence_feature)
        conv = torch.cat((conv1, conv2, conv3), 1)
        conv = self.filter_linear(conv.transpose(1, 3))
        # outputs = self.con1(sentence_feature)
        # outputs = self.max_pool(outputs)
        # outputs = self.relu(outputs)
        return conv


if __name__ == '__main__':
    model = CNN(786, 786)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    base_path = r"D:\pycharm\PyCharm 2020.1.1\workplace\machine_learning\ForNet\data"
    file_prefix = "bert_data"
    dataset = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="train")
    data_loader = DataLoader(dataset, batch_size=8, is_truncate=True)
    prefetcher = Data_Prefetcher(BatchFix(dataset_iter=data_loader))
    src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
    iteration = 0
    model.train()
    # a=torch.tensor()
    # a.int()
    embedding = nn.Embedding(21128, 786).to(device)
    while src is not None:
        iteration += 1
        # 训练代码
        # print(g.shape)
        inputs = embedding(src_id_matrix).permute(0, 3, 1, 2)
        output = model(inputs)
        print(output.shape)
        src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
