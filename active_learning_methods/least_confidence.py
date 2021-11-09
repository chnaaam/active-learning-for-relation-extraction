def least_confidence(prob_list, size):
    confidence_list = [(idx, max(prob)) for idx, prob in prob_list]
    sorted_confidence_list = list(sorted(confidence_list, key=lambda x: x[1]))

    candidate_idx_list = [idx for idx, _ in sorted_confidence_list]

    if len(candidate_idx_list) != len(list(set(candidate_idx_list))):
        raise NotImplementedError()

    if len(candidate_idx_list) > size:
        return candidate_idx_list[:size]
    else:
        return candidate_idx_list
