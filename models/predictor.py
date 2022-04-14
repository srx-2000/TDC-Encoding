"""
@ModuleName: predictor
@Description: 
@Author: Beier
@Time: 2022/4/12 12:41
"""

from my_model.models.Beam import GNMTGlobalScorer, tile
import codecs
import torch
import math
from my_model.util.metric import rouge, rouge1


def build_predictor(config_dict: dict, model):
    scorer = GNMTGlobalScorer(config_dict["alpha"], length_penalty='wu')

    translator = Translator(config_dict, model, global_scorer=scorer)
    return translator


class Translator(object):
    """
    Uses a model to translate a batch of sentences.


    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       copy_attn (bool): use copy attention during translation
       cuda (bool): use cuda
       beam_trace (bool): trace beam search for debugging
       logger(logging.Logger): logger.
    """

    def __init__(self,
                 config_dict: dict,
                 model,
                 # vocab,
                 # symbols,
                 global_scorer=None,
                 # logger=None,
                 dump_beam=""):
        self.logger = config_dict["log"]
        self.cuda = config_dict["use_gpu"]
        self.config_dict = config_dict
        # self.args = args
        self.model = model
        self.generator = self.model.generator
        self.vocab = config_dict["bertTokenizer"]
        symbols = {'BOS': self.vocab.vocab['BOS1'], 'EOS': self.vocab.vocab['EOS1'],
                   'PAD': self.vocab.vocab['[PAD]']}
        self.symbols = symbols
        self.start_token = symbols['BOS']
        self.end_token = symbols['EOS']

        self.global_scorer = global_scorer
        self.beam_size = config_dict["beam_size"]
        self.min_length = config_dict["min_length"]
        self.max_length = config_dict["max_length"]

        self.dump_beam = dump_beam

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None

        # tensorboard_log_dir = args.model_path

        # self.tensorboard_writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

    # def _build_target_tokens(self, pred):
    #     # vocab = self.fields["tgt"].vocab
    #     tokens = []
    #     for tok in pred:
    #         tok = int(tok)
    #         tokens.append(tok)
    #         if tokens[-1] == self.end_token:
    #             tokens = tokens[:-1]
    #             break
    #     tokens = [t for t in tokens if t < len(self.vocab)]
    #     tokens = self.vocab.DecodeIds(tokens).split(' ')
    #     return tokens

    def from_batch(self, translation_batch):
        batch = translation_batch["batch"]
        assert (len(translation_batch["gold_score"]) ==
                len(translation_batch["predictions"]))
        batch_size = batch[9]

        preds, pred_score, gold_score, tgt_str, src = translation_batch["predictions"], translation_batch["scores"], \
                                                      translation_batch["gold_score"], batch[3], batch[0]

        translations = []
        for b in range(batch_size):
            pred_sents = self.vocab.convert_ids_to_tokens([int(n) for n in preds[b][0]])
            pred_sents = ' '.join(pred_sents).replace(' ##', '')
            gold_sent = ' '.join(tgt_str[b].split())
            # translation = Translation(fname[b],src[:, b] if src is not None else None,
            #                           src_raw, pred_sents,
            #                           attn[b], pred_score[b], gold_sent,
            #                           gold_score[b])
            # src = self.spm.DecodeIds([int(t) for t in translation_batch['batch'].src[0][5] if int(t) != len(self.spm)])
            raw_src = [self.vocab.ids_to_tokens[int(t)] for t in src[b]][:500]
            raw_src = ' '.join(raw_src)
            translation = (pred_sents, gold_sent, raw_src)
            # translation = (pred_sents[0], gold_sent)
            translations.append(translation)

        return translations

    def translate(self,
                  data_iter, step,
                  attn_debug=False):

        self.model.eval()
        gold_path = self.config_dict["log_path"] + 'lcsts.%d.gold' % step
        can_path = self.config_dict["log_path"] + 'lcsts.%d.candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        # raw_gold_path = self.args.result_path + '.%d.raw_gold' % step
        # raw_can_path = self.args.result_path + '.%d.raw_candidate' % step
        self.gold_out_file = codecs.open(gold_path, 'w', 'utf-8')
        self.can_out_file = codecs.open(can_path, 'w', 'utf-8')

        raw_src_path = self.config_dict["log_path"] + 'lcsts.%d.raw_src' % step
        self.src_out_file = codecs.open(raw_src_path, 'w', 'utf-8')

        # pred_results, gold_results = [], []
        ct = 0
        count = 0
        with torch.no_grad():
            for batch in data_iter:
                if (self.config_dict["recall_eval"]):
                    gold_tgt_len = batch[1].size(1)
                    self.min_length = gold_tgt_len + 20
                    self.max_length = gold_tgt_len + 60
                batch_data = self.translate_batch(count, batch)
                translations = self.from_batch(batch_data)

                for trans in translations:
                    pred, gold, src = trans
                    # pred_str = pred.replace('[unused0]', '').replace('[unused3]', '').replace('[PAD]', '').replace(
                    #     '[unused1]', '').replace(r' +', ' ').replace(' [unused2] ', '<q>').replace('[unused2]',
                    #                                                                                '').strip()
                    pred_str = pred.strip()
                    gold_str = gold.strip()
                    if (self.config_dict["recall_eval"]):
                        _pred_str = ''
                        gap = 1e3
                        for sent in pred_str.split('<q>'):
                            can_pred_str = _pred_str + '<q>' + sent.strip()
                            can_gap = math.fabs(len(_pred_str.split()) - len(gold_str.split()))
                            # if(can_gap>=gap):
                            if (len(can_pred_str.split()) >= len(gold_str.split()) + 10):
                                pred_str = _pred_str
                                break
                            else:
                                gap = can_gap
                                _pred_str = can_pred_str

                        # pred_str = ' '.join(pred_str.split()[:len(gold_str.split())])
                    # self.raw_can_out_file.write(' '.join(pred).strip() + '\n')
                    # self.raw_gold_out_file.write(' '.join(gold).strip() + '\n')
                    self.can_out_file.write(pred_str + '\n')
                    self.gold_out_file.write(gold_str + '\n')
                    self.src_out_file.write(src.strip() + '\n')
                    ct += 1
                self.can_out_file.flush()
                self.gold_out_file.flush()
                self.src_out_file.flush()
                count += 1
                if count > 10:
                    count = 0
                    break
        self.can_out_file.close()
        self.gold_out_file.close()
        self.src_out_file.close()

        if (step != -1):
            rouges = self._report_rouge(gold_path, can_path)
            self.logger('Rouges at step %d \n%s' % (step, str(rouges)))
            # if self.tensorboard_writer is not None:
            #     self.tensorboard_writer.add_scalar('test/rouge1-F', rouges['rouge_1_f_score'], step)
            #     self.tensorboard_writer.add_scalar('test/rouge2-F', rouges['rouge_2_f_score'], step)
            #     self.tensorboard_writer.add_scalar('test/rougeL-F', rouges['rouge_l_f_score'], step)
            return rouges

    def _report_rouge(self, gold_path, can_path):
        self.logger("Calculating Rouge")
        # results_dict = test_rouge(self.args.temp_dir, can_path, gold_path)

        # results_dict = rouge(gold_path, can_path, self.config_dict["log_path"], self.logger)
        results_dict = rouge1(gold_path, can_path, self.logger)
        return results_dict

    def translate_batch(self, count, batch, fast=False):
        """
        Translate a batch of sentences.

        Mostly a wrapper around :obj:`Beam`.

        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
           fast (bool): enables fast beam search (may not support all features)

        Todo:
           Shouldn't need the original dataset.
        """
        with torch.no_grad():
            return self._fast_translate_batch(
                count,
                batch,
                self.max_length,
                min_length=self.min_length)

    def _fast_translate_batch(self,
                              count,
                              batch,
                              max_length,
                              min_length=0):
        # TODO: faster code path for beam_size == 1.

        # TODO: support these blacklisted features.
        assert not self.dump_beam
        device = self.config_dict["device"]
        beam_size = self.beam_size
        batch_size = batch[9]
        src = batch[0].to(device)
        segs = batch[5].to(device)
        mask_src = batch[4].to(device)
        tgt = batch[1].to(device)
        sentence_id_matrix = batch[8].to(device)
        self.model.to(device)
        src_features, attention_weight = self.model.encoder(src, tgt, segs, mask_src, sentence_id_matrix)
        dec_states = self.model.decoder.init_decoder_state(src, src_features, with_cache=True)
        # device = src_features.device

        # Tile states and memory beam_size times.
        dec_states.map_batch_fn(
            lambda state, dim: tile(state, beam_size, dim=dim))
        src_features = tile(src_features, beam_size, dim=0)
        batch_offset = torch.arange(
            batch_size, dtype=torch.long, device=device)
        beam_offset = torch.arange(
            0,
            batch_size * beam_size,
            step=beam_size,
            dtype=torch.long,
            device=device)
        alive_seq = torch.full(
            [batch_size * beam_size, 1],
            self.start_token,
            dtype=torch.long,
            device=device)

        # Give full probability to the first beam on the first step.
        topk_log_probs = (
            torch.tensor([0.0] + [float("-inf")] * (beam_size - 1),
                         device=device).repeat(batch_size))

        # Structure that holds finished hypotheses.
        hypotheses = [[] for _ in range(batch_size)]  # noqa: F812

        results = {}
        results["predictions"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["scores"] = [[] for _ in range(batch_size)]  # noqa: F812
        results["gold_score"] = [0] * batch_size
        results["batch"] = batch

        for step in range(max_length):
            print(f"\r正在生成摘要，进行到{count}轮{step}步，共{max_length}步", end="", flush=True)
            decoder_input = alive_seq[:, -1].view(1, -1)

            # Decoder forward.
            decoder_input = decoder_input.transpose(0, 1)

            dec_out, dec_states = self.model.decoder(decoder_input, src_features, dec_states,
                                                     step=step)

            # Generator forward.
            log_probs = self.generator.forward(dec_out.transpose(0, 1).squeeze(0))
            vocab_size = log_probs.size(-1)

            if step < min_length:
                log_probs[:, self.end_token] = -1e20

            # Multiply probs by the beam probability.
            log_probs += topk_log_probs.view(-1).unsqueeze(1)

            alpha = self.global_scorer.alpha
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha

            # Flatten probs into a list of possibilities.
            curr_scores = log_probs / length_penalty

            if (self.config_dict["block_trigram"]):
                cur_len = alive_seq.size(1)
                if (cur_len > 3):
                    for i in range(alive_seq.size(0)):
                        fail = False
                        words = [int(w) for w in alive_seq[i]]
                        words = [self.vocab.ids_to_tokens[w] for w in words]
                        words = ' '.join(words).replace(' ##', '').split()
                        if (len(words) <= 3):
                            continue
                        trigrams = [(words[i - 1], words[i], words[i + 1]) for i in range(1, len(words) - 1)]
                        trigram = tuple(trigrams[-1])
                        if trigram in trigrams[:-1]:
                            fail = True
                        if fail:
                            curr_scores[i] = -10e20

            curr_scores = curr_scores.reshape(-1, beam_size * vocab_size)
            topk_scores, topk_ids = curr_scores.topk(beam_size, dim=-1)

            # Recover log probs.
            topk_log_probs = topk_scores * length_penalty

            # Resolve beam origin and true word ids.
            topk_beam_index = topk_ids.div(vocab_size)
            topk_ids = topk_ids.fmod(vocab_size)

            # Map beam_index to batch_index in the flat representation.
            batch_index = (
                    topk_beam_index
                    + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
            select_indices = batch_index.view(-1)

            # Append last prediction.
            alive_seq = torch.cat(
                [alive_seq.index_select(0, select_indices),
                 topk_ids.view(-1, 1)], -1)

            is_finished = topk_ids.eq(self.end_token)
            if step + 1 == max_length:
                is_finished.fill_(1)
            # End condition is top beam is finished.
            end_condition = is_finished[:, 0].eq(1)
            # Save finished hypotheses.
            if is_finished.any():
                predictions = alive_seq.view(-1, beam_size, alive_seq.size(-1))
                for i in range(is_finished.size(0)):
                    b = batch_offset[i]
                    if end_condition[i]:
                        is_finished[i].fill_(1)
                    finished_hyp = is_finished[i].nonzero().view(-1)
                    # Store finished hypotheses for this batch.
                    for j in finished_hyp:
                        hypotheses[b].append((
                            topk_scores[i, j],
                            predictions[i, j, 1:]))
                    # If the batch reached the end, save the n_best hypotheses.
                    if end_condition[i]:
                        best_hyp = sorted(
                            hypotheses[b], key=lambda x: x[0], reverse=True)
                        score, pred = best_hyp[0]

                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                non_finished = end_condition.eq(0).nonzero().view(-1)
                # If all sentences are translated, no need to go further.
                if len(non_finished) == 0:
                    break
                # Remove finished batches for the next step.
                topk_log_probs = topk_log_probs.index_select(0, non_finished)
                batch_index = batch_index.index_select(0, non_finished)
                batch_offset = batch_offset.index_select(0, non_finished)
                alive_seq = predictions.index_select(0, non_finished) \
                    .view(-1, alive_seq.size(-1))
            # Reorder states.
            select_indices = batch_index.view(-1)
            src_features = src_features.index_select(0, select_indices)
            dec_states.map_batch_fn(
                lambda state, dim: state.index_select(dim, select_indices))

        return results
