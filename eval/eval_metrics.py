import numpy as np
from utils import preprocess


def precision_at_k(relevance_score, k):
    """ Precision at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.mean(relevance_score)


def recall_at_k(relevance_score, k, m):
    """ Recall at K given binary relevance scores. """
    assert k >= 1
    relevance_score = np.asarray(relevance_score)[:k] != 0
    if relevance_score.size != k:
        raise ValueError('Relevance score length < K')
    return np.sum(relevance_score) / float(m)


def mean_precision_at_k(relevance_scores, k):
    """ Mean Precision at K given binary relevance scores. """
    mean_p_at_k = np.mean(
        [precision_at_k(r, k) for r in relevance_scores]).astype(np.float32)
    return mean_p_at_k


def mean_recall_at_k(relevance_scores, k, m_list):
    """ Mean Recall at K:  m_list is a list containing # relevant target entities for each data point. """
    mean_r_at_k = np.mean([recall_at_k(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return mean_r_at_k


def average_precision(relevance_score, K, m):
    """ For average precision, we use K as input since the number of prediction targets is not fixed
    unlike standard IR evaluation. """
    r = np.asarray(relevance_score) != 0
    out = [precision_at_k(r, k + 1) for k in range(0, K) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / float(min(K, m))


def MAP(relevance_scores, k, m_list):
    """ Mean Average Precision -- MAP. """
    map_val = np.mean([average_precision(r, k, M) for r, M in zip(relevance_scores, m_list)]).astype(np.float32)
    return map_val


def MRR(relevance_scores):
    """ Mean reciprocal rank -- MRR. """
    rs = (np.asarray(r).nonzero()[0] for r in relevance_scores)
    mrr_val = np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs]).astype(np.float32)
    return mrr_val


def get_masks(top_k, inputs):
    """ Mask the dummy sequences  -- : 0 if .. 1 if seed set is of size > 1. """
    masks = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        if len(seeds) == 1 and list(seeds)[0] == preprocess.start_token:
            masks.append(0)
        else:
            masks.append(1)
    return np.array(masks).astype(np.int32)


def remove_seeds(top_k, inputs):
    """ Replace seed users from top-k predictions with -1. """
    result = []
    for i in range(0, top_k.shape[0]):
        seeds = set(inputs[i])
        lst = list(top_k[i])  # top-k predicted users.
        for s in seeds:
            if s in lst:
                lst.remove(s)
        for k in range(len(top_k[i]) - len(lst)):
            lst.append(-1)
        result.append(lst)
    return np.array(result).astype(np.int32)


def get_relevance_scores(top_k_filter, targets):
    """ Create binary relevance scores by checking if the top-k predicted users are in target set. """
    output = []
    for i in range(0, top_k_filter.shape[0]):
        z = np.isin(top_k_filter[i], targets[i])
        output.append(z)
    return np.array(output)
