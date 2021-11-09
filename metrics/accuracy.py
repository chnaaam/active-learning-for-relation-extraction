def calc_acc(true_y, pred_y):
    total = 0
    acc = 0

    for ty, py in zip(true_y, pred_y):
        if ty == py:
            acc += 1

        total += 1

    return acc / total
