"""
@ModuleName: generator
@Description: 
@Author: Beier
@Time: 2022/4/6 13:16
"""

from torch import nn
import torch
from my_model.models.encoder import Encoder
from my_model.models.decoder import TransformerDecoder
from my_model.util.dataLoader import DataSetMultiFile, Data_Prefetcher, DataLoader, BatchFix


# from my_model.models.Beam import Beam


# loss = nn.CrossEntropyLoss(ignore_index=utils.PAD, reduction='none')
def get_generator(vocab_size, dec_hidden_size, device):
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(dec_hidden_size, vocab_size),
        gen_func
    )
    generator.to(device)

    return generator


class Generator(nn.Module):
    def __init__(self, config_dict: dict):
        # num_layers, d_model, heads, d_ff, embedding_size, dropout = 0.1, is_share_emb = False
        super(Generator, self).__init__()
        self.encoder = Encoder(config_dict)
        # self.embedding_size = config_dict["embedding_size"]
        self.tgt_embedding = nn.Embedding(self.encoder.config.vocab_size, self.encoder.config.hidden_size,
                                          padding_idx=0)
        self.final_linear = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.vocab_size)
        self.soft_max = nn.Softmax(-1)
        # if self.embedding_size != self.encoder.config.hidden_size:
        #     self.embedding_mapping_lin = nn.Linear(self.embedding_size, self.encoder.config.hidden_size)
        # if is_share_emb:
        #     self.encoder.
        self.decoder = TransformerDecoder(num_layers=config_dict["num_layers"], d_model=config_dict["d_model"],
                                          heads=config_dict["heads"], d_ff=config_dict["d_ff"],
                                          dropout=config_dict["dropout"], embeddings=self.tgt_embedding)
        self.is_fuse = config_dict["is_fuse"]
        self.config_dict = config_dict
        self.tokenizer = config_dict["bertTokenizer"]
        self.generator = get_generator(self.encoder.config.vocab_size, config_dict["d_model"], config_dict["device"])
        self.generator[0].weight = self.decoder.embeddings.weight

    def forward(self, src, tgt, segs, mask_src, sentence_id_matrix):

        output, attention_weight = self.encoder(src, tgt, segs, mask_src, sentence_id_matrix)
        if self.is_fuse:
            dec_state = self.decoder.init_decoder_state(src, output)
        else:
            dec_state = self.decoder.init_decoder_state(sentence_id_matrix.flatten(1, 2), output)
        decoder_outputs, state = self.decoder(tgt, output, dec_state)
        # logic = self.soft_max(self.final_linear(decoder_outputs))
        # return logic.view(-1, logic.size(-1)), None
        decoder_outputs = self.final_linear(decoder_outputs).transpose(1, 0)
        return decoder_outputs, None

    def sample(self, src, tgt, segs, mask_src, sentence_id_matrix):
        bos = torch.ones(tgt.shape).long().fill_(self.tokenizer.cls_token_id)
        if self.config_dict["use_gpu"]:
            bos = bos.to(self.config_dict["device"])
        output, attention_weight = self.encoder(src, tgt, segs, mask_src, sentence_id_matrix)
        if self.is_fuse:
            state = self.decoder.init_decoder_state(src, output)
        else:
            state = self.decoder.init_decoder_state(sentence_id_matrix.flatten(1, 2), output)
        inputs, outputs = [bos], []
        for i in range(self.config_dict["max_time_step"]):
            output_dec, state = self.decoder(inputs[i], output, state)
            output_dec = self.final_linear(output_dec).transpose(1, 0)
            predicted = output_dec.max(2)[1]
            inputs += [predicted.transpose(0, 1).contiguous()]
            outputs += [predicted]
        outputs = torch.stack(outputs)
        # print(outputs)
        return outputs
        # return decoder_outputs, None


if __name__ == '__main__':
    model = Generator()
    # model = Encoder()
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
        output = model(src=src, tgt=tgt, segs=segs, mask_src=mask, sentence_id_matrix=src_id_matrix, is_fuse=True)
        print(output[0].shape)
        src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
