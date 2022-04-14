"""
@ModuleName: encoder
@Description: 
@Author: Beier
@Time: 2022/4/4 22:09
"""
import torch
from torch import nn
from my_model.models.CNN import CNN
from my_model.models.Bert import Bert
from my_model.models.neural import vocab_map_sentence, luong_gate_attention
from my_model.util.dataLoader import DataSetMultiFile, Data_Prefetcher, DataLoader, BatchFix


class Encoder(nn.Module):
    def __init__(self, config_dict: dict, stride=1):
        super(Encoder, self).__init__()
        self.bert = Bert(config_dict)
        self.config = self.bert.bert_config
        self.sentence_feature = self.bert.sentence_feature
        self.embedding_mapping_lin = self.bert.embedding_mapping_lin if hasattr(self.bert,
                                                                                "embedding_mapping_lin") else None
        self.cnn = CNN(stack_channel=self.config.hidden_size, output_channel=self.config.hidden_size,
                       stride=stride)
        self.lga = luong_gate_attention(self.config.hidden_size)
        self.is_fuse = config_dict["is_fuse"]

    def forward(self, src, tgt, seg, src_mask, sentence_id_matrix):
        query = self.bert(src, tgt, seg, src_mask,
                          sentence_id_matrix).float()  # Batch_size * seg_num * short_seg_len * sentence_len
        if not self.embedding_mapping_lin:
            inputs = self.sentence_feature(sentence_id_matrix).permute(0, 3, 1,
                                                                       2).contiguous()  # Batch_size * feature * seg_num * short_seg_len
        else:
            inputs = self.embedding_mapping_lin(self.sentence_feature(sentence_id_matrix)).permute(0, 3, 1,
                                                                                                   2).contiguous()
        key_value = self.cnn(inputs)
        output, weights = self.lga(query, key_value, self.is_fuse)
        return output, weights


if __name__ == '__main__':
    model = Encoder()
    # model = Bert()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    base_path = r"D:\pycharm\PyCharm 2020.1.1\workplace\machine_learning\ForNet\data"
    file_prefix = "bert_data"
    dataset = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="train")
    data_loader = DataLoader(dataset, batch_size=2, is_truncate=True)
    prefetcher = Data_Prefetcher(BatchFix(dataset_iter=data_loader))
    src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
    iteration = 0
    model.train()
    while src is not None:
        iteration += 1
        # 训练代码
        # print(g.shape)
        # inputs = embedding(src_id_matrix.int()).permute(0, 3, 1, 2)
        output = model(src=src, tgt=tgt, seg=segs, src_mask=mask, sentence_id_matrix=src_id_matrix)
        print(output[0].shape)
        src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
    # print(encoder)
    # tensor = torch.ones(12).reshape((3, 4))
    # print(tensor.shape[0])
