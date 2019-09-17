# First version of ML, h(x) = a0 * x0 + a1 * x1 + a3 * x3
#
import numpy as np
import math
raw_data = [(2, 4, 8),
            (5, 3, 13),
            (3, 5, 11),
            (9, 7, 25)]

learning_rate = 0.000001
training_iteration = 1790000

weight = np.ones(len(raw_data[0]))
gradient = []
last_total_loss = 0


def forward(record):
    predict = weight[0] * 1  # calculate the first weight is present
    for j in range(len(record) - 1):  # the last record is target, so minus 1 here
        predict = predict + weight[j + 1] * record[j]
    return predict


def loss_function():
    total_loss = sum([math.pow(compute_record_loss(record), 2) for record in raw_data])
    return 0.5 * total_loss


def gradient_j(j):
    if j == 0:
        return sum([compute_record_loss(record) for record in raw_data])
    else:
        return sum([compute_record_loss(record) * record[j] for record in raw_data])


def compute_record_loss(record):
    label = record[-1]
    return label - forward(record)


def train():
    global last_total_loss
    for step in range(training_iteration):
        total_loss = loss_function()
        print("Iteration %d weight %s, total loss: %f" % (step, str(weight), total_loss))

        # update weight
        for j in range(len(weight)):
            weight[j] = weight[j] + learning_rate * gradient_j(j)

        if early_stop_check(step, total_loss):
            print("Early stop triggered, stop at iteration %d with loss function value %f" % (step, total_loss))
            break
        else:
            last_total_loss = total_loss


def early_stop_check(step, current_loss):
    if step < 20:
        return False
    return current_loss >= last_total_loss


train()
print("final weight: %s" % (",".join([str(w) for w in weight])))

print("9 * 11 = %d" % forward((9, 11, 0)))

