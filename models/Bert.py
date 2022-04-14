"""
@ModuleName: Bert
@Description: 
@Author: Beier
@Time: 2022/4/4 18:56
"""
import torch
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer
from my_model.util.dataLoader import DataLoader, DataSetMultiFile, Data_Prefetcher, BatchFix
from my_model.models.neural import vocab_map_sentence
from my_model.models.CNN import CNN
from transformers import logging

#
# logging.set_verbosity_error()
# logging.set_verbosity_warning()

new_tokens = ["BOS1", "EOS1"]


class Bert(nn.Module):
    def __init__(self, config_dict: dict):
        super(Bert, self).__init__()
        self.embedding_size = config_dict["embedding_size"]
        if config_dict["pre_train_model"] == "" or config_dict["pre_train_model_config"] == "":
            self.bert_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=r"../../temp")
            self.bert_config = BertConfig.from_pretrained("hfl/chinese-roberta-wwm-ext", cache_dir=r"../../temp")
        else:
            self.bert_model = BertModel.from_pretrained(config_dict["pre_train_model"], cache_dir=r"../../temp")
            self.bert_config = BertConfig.from_pretrained(config_dict["pre_train_model_config"],
                                                          cache_dir=r"../../temp")
        # tokenizer = BertTokenizer(vocab_file=r"C:\Users\16016\Desktop\vocab.txt")
        # tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        # self.bert_model.resize_token_embeddings(len(tokenizer))
        self.mapping_linear = vocab_map_sentence(self.bert_config.hidden_size)
        self.sentence_feature = nn.Embedding(self.bert_config.vocab_size, self.embedding_size, padding_idx=0)
        if self.embedding_size != self.bert_config.hidden_size:
            self.embedding_mapping_lin = nn.Linear(self.embedding_size, self.bert_config.hidden_size)

    def forward(self, src, tgt, seg, src_mask, sentence_id_matrix):
        outputs = self.bert_model(src, seg, src_mask)
        embedding = self.sentence_feature(sentence_id_matrix)
        if self.embedding_size != self.bert_config.hidden_size:
            embedding = self.embedding_mapping_lin(embedding)
        self.mapping_linear.init_context(embedding)
        outputs = self.mapping_linear(outputs[0])
        return outputs


if __name__ == '__main__':
    model = Bert()
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
    # embedding = nn.Embedding(21128, 786).to(device)
    while src is not None:
        iteration += 1
        # 训练代码
        # print(g.shape)
        # inputs = embedding(src_id_matrix.int()).permute(0, 3, 1, 2)
        output = model(src, tgt, segs, mask, src_id_matrix)
        print(src_id_matrix.shape)
        print(output.shape)
        # print(output[1].shape)
        src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
