"""
@ModuleName: main
@Description: 
@Author: Beier
@Time: 2022/4/11 9:40
"""

from my_model.util.dataLoader import DataSetMultiFile, Data_Prefetcher, DataLoader, BatchFix
from my_model.models.generator import Generator
import torch
from torch import nn
from transformers import BertTokenizer
from torch import optim
import time
import math
import os
import yaml
from my_model.models.predictor import build_predictor
import glob
import codecs

config_path = os.path.join(os.path.dirname(os.getcwd()), "config.ymal")
config_file = open(config_path, mode="r", encoding="utf-8")
config_dict = yaml.load(config_file)
# print(config_dict)
tokenizer = BertTokenizer(vocab_file=config_dict["pre_train_tokenizer"])
config_dict["bertTokenizer"] = tokenizer

eval_interval = config_dict["eval_interval"]
logF = config_dict["logF"]
log_dir = config_dict["log_dir"]
save_interval = config_dict["save_interval"]


def print_logs(file):
    def write_log(s):
        print(s, end='')
        with open(file, 'a') as f:
            f.write(s)

    return write_log


# log
def build_log():
    # log
    if not os.path.exists(logF):
        os.mkdir(logF)
    log_path = logF + log_dir + os.sep
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    print_log = print_logs(log_path + 'log.txt')
    return print_log, log_path


def progress_bar(percent: float, length: int, start_str: str, end_str: str, r=2):
    current_count = int(percent * length)
    unfinished = "." * (length - current_count)
    finished = "=" * current_count + ">"
    base_str = "\r" + f"{start_str}: [" + finished + unfinished + f"] {round(percent * 100, r)}% {end_str}"
    print(base_str, end="", flush=True)


def train(model, data, optimizer, criterion, epoch, params, device, use_gan=False):
    iterator = data["train_iter"]
    if use_gan:
        pass
    else:
        # model = Generator()
        model.train()
        epoch_loss = 0
        # device = "cpu"
        model.to(device)
        # base_path = r"D:\pycharm\PyCharm 2020.1.1\workplace\machine_learning\ForNet\data"
        # file_prefix = "bert_data"
        # dataset = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="train")
        # data_loader = DataLoader(dataset, batch_size=8, is_truncate=True)
        # prefetcher = Data_Prefetcher(BatchFix(dataset_iter=data_loader))
        # src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
        iteration = 0
        checkpoint_step = 0
        # while src is not None:
        for step, (src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix, batch_size) in enumerate(
                iterator):
            # progress_bar()
            src = src.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)
            segs = segs.to(device)
            src_id_matrix = src_id_matrix.to(device)
            iteration += 1
            optimizer.zero_grad()
            output = model(src=src, tgt=tgt, segs=segs, mask_src=mask, sentence_id_matrix=src_id_matrix)[0].contiguous()
            pred = output.max(2)[1]
            targets = tgt.t()
            num_correct = pred.eq(targets).masked_select(targets.ne(tokenizer.pad_token_id)).sum().item()
            num_total = targets.ne(tokenizer.pad_token_id).sum().item()
            output_dim = output[0].shape[-1]
            sum = tgt.transpose(1, 0).contiguous()
            outputs = output[1:].view(-1, output_dim)
            sum = sum[1:].view(-1)

            # sum = [(sum len - 1) * batch size]
            # output = [(sum len - 1) * batch size, output dim]

            loss = criterion(outputs, sum)
            loss = torch.sum(loss) / num_total
            loss.backward()
            optimizer.step()
            params['updates'] += 1

            params['report_loss'] += loss.item()
            params['report_correct'] += num_correct
            params['report_total'] += num_total
            print(f"\r第{params['updates']}步：loss:{params['report_loss']}", end="", flush=True)
            if params["updates"] % eval_interval == 0:
                params['log']("\n epoch: %3d, loss: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                              % (epoch, params['report_loss'], time.time() - params['report_time'],
                                 params['updates'], params['report_correct'] * 100.0 / params['report_total']))
                print('evaluating after %d updates...\r' % params['updates'])
                score = evaluate(model, data, params, device, criterion)

                params["rouge"].append(score)
                if score >= max(params["rouge"]):
                    with codecs.open(params['log_path'] + 'best_rouge_prediction.txt', 'w', 'utf-8') as f:
                        f.write(
                            codecs.open(params['log_path'] + f'lcsts.{checkpoint_step}.candidate', 'r', 'utf-8').read())
                    save_model(params['log_path'] + 'best_rouge_checkpoint.pt', model, optimizer,
                               params['updates'])
                model.train()
                params['report_loss'], params['report_time'] = 0, time.time()
                params['report_correct'], params['report_total'] = 0, 0
            # progress_bar(iteration / num_data_set,length=50)
            # epoch_loss += loss.item()
            if params["updates"] % save_interval == 0:
                checkpoint_step = params["updates"]
                save_model(params['log_path'] + f'checkpoint_step_{params["updates"]}.pt', model, optimizer,
                           params['updates'])
            # src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix = prefetcher.next()
        # return epoch_loss / iteration


def evaluate(model, data, params, device, criterion):
    model.eval()
    ### 不启用 BatchNormalization 和 Dropout，
    ### 评估/验证的常用做法是torch.no_grad()与配对使用model.eval()以关闭梯度计算：
    epoch_loss = 0
    iterator = data["val_iter"]
    _log_path = params["log_path"]
    cp_files = sorted(glob.glob(os.path.join(_log_path, 'checkpoint_step_*.pt')))
    cp_files.sort(key=os.path.getmtime)
    if (cp_files):
        cp = cp_files[-1]
        step = int(cp.split('.')[-2].split('_')[-1])
    else:
        step = 0
    with torch.no_grad():
        iteration = 0
        predictor = build_predictor(config_dict, model)
        scores = predictor.translate(iterator, step)
        losss = 0
        for step, (src, tgt, src_txt, tgt_txt, mask, segs, clss, src_matrix, src_id_matrix, batch_size) in enumerate(
                iterator):
            iteration += 1
            src = src.to(device)
            tgt = tgt.to(device)
            mask = mask.to(device)
            segs = segs.to(device)
            src_id_matrix = src_id_matrix.to(device)
            output = model(src, tgt, mask, segs, src_id_matrix)[0].contiguous()

            # pred = output.max(2)[1]
            targets = tgt.t()
            # num_correct = pred.eq(targets).masked_select(targets.ne(tokenizer.pad_token_id)).sum().item()
            num_total = targets.ne(tokenizer.pad_token_id).sum().item()
            output_dim = output[0].shape[-1]
            sum = tgt.transpose(1, 0).contiguous()
            outputs = output[1:].view(-1, output_dim)
            sum = sum[1:].view(-1)
            loss = criterion(outputs, sum)
            loss = torch.sum(loss) / num_total
            losss += loss
            print(f"\r{loss}", end="", flush=True)
        print_log(f"\r验证集loss:{losss}")
        #
        #     # output = model(doc, doc_len, sum, 0)  # 验证时不使用teacher forcing
        #     # model.sample(src=src, tgt=tgt, segs=segs, mask_src=mask, sentence_id_matrix=src_id_matrix)
        #     # sum = [sum len, batch size]
        #     # output = [sum len, batch size, output dim]

        #
        #     output_dim = output.shape[-1]
        #
        #     sum = tgt.transpose(1, 0).contiguous()
        #     output = output[1:].view(-1, output_dim)
        #     sum = sum[1:].view(-1)
        #
        #     # sum = [(sum len - 1) * batch size]
        #     # output = [(sum len - 1) * batch size, output dim]
        #
        #     loss = criterion(output, sum)
        #
        #     epoch_loss += loss.item()
        #     if step > 20: break

    return scores


def save_model(path, model, optim, updates):
    model_state_dict = model.state_dict()
    checkpoint = {'model': model_state_dict,
                  'optim': optim,
                  'updates': updates}
    torch.save(checkpoint, path)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_dict["use_gpu"] = torch.cuda.is_available()
    config_dict["device"] = device
    base_path = config_dict["data_base_path"]
    file_prefix = config_dict["data_file_prefix"]
    dataset_train = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="train")
    dataset_valid = DataSetMultiFile(base_path, file_prefix, is_shuffle=True, mode="valid")
    data_loader_train = DataLoader(dataset_train, batch_size=config_dict["batch_size"],
                                   is_truncate=config_dict["is_truncate"])
    data_loader_valid = DataLoader(dataset_valid, batch_size=1,
                                   is_truncate=config_dict["is_truncate"])
    train_iter = iter(BatchFix(dataset_iter=data_loader_train))
    val_iter = iter(BatchFix(dataset_iter=data_loader_valid))
    data = {"train_iter": train_iter, "val_iter": val_iter}
    model = Generator(config_dict)
    N_EPOCHS = config_dict["epoch"]
    lr = config_dict["lr"]
    weight_decay = config_dict["weight_decay"]
    # 使用ignore_index参数，使得计算损失的时候不计算pad的损失
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    print_log, log_path = build_log()
    params = {'updates': 0, 'report_loss': 0, 'report_total': 0,
              'report_correct': 0, 'report_time': time.time(),
              'log': print_log, 'log_path': log_path, 'rouge': []}
    config_dict["log"] = print_log
    config_dict["log_path"] = log_path
    # params["rouge"] = []
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        # train_loss = train(model, data, optimizer, criterion, epoch, params, device)
        # valid_loss = evaluate(model, data, device, criterion)
        train(model, data, optimizer, criterion, epoch, params, device)
        # evaluate(model, data, device, criterion)

        # end_time = time.time()
        #
        # epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        #
        # print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        # print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
