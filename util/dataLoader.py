"""
@ModuleName: dataLoader
@Description: 
@Author: Beier
@Time: 2022/3/25 16:33
"""
import torch.utils.data as torch_dataset
import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import RandomSampler
import random
import os
import glob
import gc


class Data_Prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream(device="cuda")
        self.preload()

    def preload(self):
        try:
            self.src, self.tgt, self.src_txt, self.tgt_txt, self.mask, self.segs, self.clss, self.src_matrix, self.src_id_matrix = next(
                self.loader)
        except StopIteration:
            self.src = None
            self.tgt = None
            self.src_txt = None
            self.tgt_txt = None
            self.mask = None
            self.segs = None
            self.clss = None
            self.src_matrix = None
            self.src_id_matrix = None
            return
        with torch.cuda.stream(self.stream):
            self.src = self.src.cuda(non_blocking=True)
            self.tgt = self.tgt.cuda(non_blocking=True)
            # self.src_txt = self.src_txt.cuda(non_blocking=True)
            # self.tgt_txt = self.tgt_txt.cuda(non_blocking=True)
            self.mask = self.mask.cuda(non_blocking=True)
            self.segs = self.segs.cuda(non_blocking=True)
            # self.clss = self.clss.cuda(non_blocking=True)
            # self.src_matrix = self.src_matrix.cuda(non_blocking=True)
            self.src_id_matrix = self.src_id_matrix.cuda(non_blocking=True)
            # self.src = self.src.float()
            # self.tgt = self.tgt.float()
            # self.mask = self.mask.float()
            # self.segs = self.segs.float()
            # self.src_id_matrix = self.src_id_matrix.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        src = self.src
        tgt = self.tgt
        src_txt = self.src_txt
        tgt_txt = self.tgt_txt
        mask = self.mask
        segs = self.segs
        clss = self.clss
        src_matrix = self.src_matrix
        src_id_matrix = self.src_id_matrix
        self.preload()
        return src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix


class DataSetMultiFile(object):
    def __init__(self, base_path, file_prefix, is_shuffle=True, mode="train"):
        assert mode in ["train", "valid", "test"]
        file_path = os.path.join(base_path, file_prefix)
        self.mode = mode
        self.is_shuffle = is_shuffle
        self.pts = sorted(glob.glob(file_path + '.[0-9]*.' + self.mode + '.pt'))
        if self.pts:
            if is_shuffle:
                random.shuffle(self.pts)
        else:
            # Only one inputters.*Dataset, simple!
            self.pt = file_path + '.' + mode + '.pt'

    def __iter__(self):
        if self.pts:
            for pt in self.pts:
                data = torch.load(pt)
                print(f"加载{self.mode}数据中,加载文件为：{pt}，数据集大小为：{len(data)}")
                yield data
                # yield _lazy_load_data(pt, self.model)
        else:
            data = torch.load(self.pt)
            print(f"加载{self.mode}数据中,加载文件为：{self.pt}，数据集大小为：{len(data)}")
            yield data


class DataLoader:
    def __init__(self, dataset, batch_size, is_truncate=True):

        self.dataset_iter = iter(dataset)
        self.batch_size = batch_size
        self.is_truncate = is_truncate
        self.cur_data = next(self.dataset_iter)
        self.count = 0

    def __len__(self):
        length = len(self.cur_data)
        from math import ceil
        if self.is_truncate:
            length = length // self.batch_size
        else:
            length = ceil(length / self.batch_size)
        return length

    def __iter__(self):
        while self.cur_data is not None:
            length = self.__len__()
            for _ in range(length):
                if length > self.count:
                    yield self.cur_data[self.count * self.batch_size:(self.count + 1) * self.batch_size]
                    self.count += 1
                else:
                    self.count = 0
            try:
                self.cur_data = next(self.dataset_iter)
            except StopIteration:
                self.cur_data = None


class BatchFix(object):
    def __init__(self, dataset_iter):
        self.data_iter = iter(dataset_iter)
        self.batch_size = dataset_iter.batch_size

    def __iter__(self):
        for batch_data in self.data_iter:
            src = []
            tgt = []
            segs = []
            mask = []
            src_txt = []
            tgt_txt = []
            clss = []
            src_matrix = []
            src_id_matrix = []
            for data in batch_data:
                src.append(data["src"])
                tgt.append(data["tgt"])
                segs.append(data["segs"])
                src_txt.append(data["src_txt"])
                tgt_txt.append(data["tgt_txt"])
                clss.append(data["clss"])
                mask.append(data["mask"])
                src_matrix.append(data["src_matrix"])
                src_id_matrix.append(data["src_id_matrix"])
            yield torch.tensor(src), torch.tensor(tgt), src_txt, tgt_txt, torch.tensor(mask), torch.tensor(
                segs), clss, src_matrix, torch.tensor(src_id_matrix),self.batch_size


if __name__ == '__main__':
    # data = torch.load(r"D:\pycharm\PyCharm 2020.1.1\workplace\machine_learning\ForNet\data\bert_data.0.test.pt")
    # print(data)
    base_path = r"D:\pycharm\PyCharm 2020.1.1\workplace\machine_learning\ForNet\data"
    file_prefix = "bert_data"
    dataset = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="train")
    mdl = DataLoader(dataset, batch_size=8, is_truncate=True)
    bf = BatchFix(mdl)
    prefetcher = Data_Prefetcher(bf)
    data, label, a, b, c, d, e, f, g = prefetcher.next()
    iteration = 0
    while data is not None:
        iteration += 1
        # 训练代码
        print(g.shape)
        data, label, a, b, c, d, e, f, g = prefetcher.next()
