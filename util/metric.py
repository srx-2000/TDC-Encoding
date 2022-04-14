"""
@ModuleName: metric
@Description: 
@Author: Beier
@Time: 2022/4/12 9:17
"""
from rouge import Rouge
import os
import codecs


def rouge(reference, candidate, log_path, print_log):
    assert len(reference) == len(candidate)

    ref_dir = log_path + 'reference' + os.sep
    cand_dir = log_path + 'candidate' + os.sep
    if not os.path.exists(ref_dir):
        os.mkdir(ref_dir)
    if not os.path.exists(cand_dir):
        os.mkdir(cand_dir)

    for i in range(len(reference)):
        with codecs.open(ref_dir + "%06d_reference.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(reference[i]).replace(' <\s> ', '\n') + '\n')
        with codecs.open(cand_dir + "%06d_candidate.txt" % i, 'w', 'utf-8') as f:
            f.write(" ".join(candidate[i]).replace(' <\s> ', '\n').replace('<unk>', 'UNK') + '\n')
    rouge = Rouge()
    for i in range(len(reference)):
        cand_name = "%06d_candidate.txt" % i
        ref_name = "%06d_reference.txt" % i
        cand_dir_fin = os.path.join(cand_dir, cand_name)
        ref_dir_fin = os.path.join(ref_dir, ref_name)
        cand = open(cand_dir_fin, encoding="utf-8").readlines()
        ref = open(ref_dir_fin, encoding="utf-8").readlines()
        scores = rouge.get_scores(cand, ref)

        recall1 = [round(scores[0]["rouge-1"]["r"] * 100, 2),
                   round(scores[0]["rouge-2"]["r"] * 100, 2),
                   round(scores[0]["rouge-l"]["r"] * 100, 2)]
        precision1 = [round(scores[0]["rouge-1"]["p"] * 100, 2),
                      round(scores[0]["rouge-2"]["p"] * 100, 2),
                      round(scores[0]["rouge-l"]["p"] * 100, 2)]
        f_score1 = [round(scores[0]["rouge-1"]["f"] * 100, 2),
                    round(scores[0]["rouge-2"]["f"] * 100, 2),
                    round(scores[0]["rouge-l"]["f"] * 100, 2)]
        print_log("F_measure: %s Recall: %s Precision: %s\n"
                  % (str(f_score1), str(recall1), str(precision1)))
        return f_score1[:], recall1[:], precision1[:]


def rouge1(reference_path, candidate_path, print_log):
    rouge = Rouge()
    cand = open(reference_path, encoding="utf-8").readlines()
    ref = open(candidate_path, encoding="utf-8").readlines()
    for i in range(len(cand)):
        scores = rouge.get_scores(cand[i], ref[i])
        recall1 = [round(scores[0]["rouge-1"]["r"] * 100, 2),
                   round(scores[0]["rouge-2"]["r"] * 100, 2),
                   round(scores[0]["rouge-l"]["r"] * 100, 2)]
        precision1 = [round(scores[0]["rouge-1"]["p"] * 100, 2),
                      round(scores[0]["rouge-2"]["p"] * 100, 2),
                      round(scores[0]["rouge-l"]["p"] * 100, 2)]
        f_score1 = [round(scores[0]["rouge-1"]["f"] * 100, 2),
                    round(scores[0]["rouge-2"]["f"] * 100, 2),
                    round(scores[0]["rouge-l"]["f"] * 100, 2)]
        print_log("F_measure: %s Recall: %s Precision: %s\n"
                  % (str(f_score1), str(recall1), str(precision1)))
        return f_score1[:], recall1[:], precision1[:]
