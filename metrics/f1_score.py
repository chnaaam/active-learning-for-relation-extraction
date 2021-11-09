from seqeval.metrics import f1_score
from seqeval.scheme import IOBES


def calc_f1_score(true_y, pred_y, mode="strict", scheme=IOBES):
    return f1_score(true_y, pred_y, mode=mode, scheme=scheme)
