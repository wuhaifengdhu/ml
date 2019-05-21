# First version of ML, h(x) = a0 * x0 + a1 * x1 + a3 * x3
#
import numpy as np
import math
raw_data = [(1.01, 0.99, 2.00), (2.0, 2.01, 4.00), (3.01, 2.999, 6.00), (4.0005, 3.999, 8.00)]


learning_rate = 0.01
training_iteration = 210000
early_stop = 0.0003

weight = np.ones(len(raw_data[0]))
gradient = []


def compute_record_loss(record):
    record_loss = weight[0] * 1  # calculate the first weight is present
    for i in range(len(record) - 1):  # the last record is target, so minus 1 here
        record_loss = record_loss + weight[i + 1] * record[i]
    record_loss = record_loss - record[-1]
    return record_loss


def loss_function():
    total_loss = sum([math.pow(compute_record_loss(record), 2) for record in raw_data])
    return total_loss * 0.5


def gradient_i(i):
    if i == 0:
        return sum([compute_record_loss(record) for record in raw_data])
    else:
        return sum([compute_record_loss(record) * record[i] for record in raw_data])


def train():
    for step in range(training_iteration):
        gradients = [gradient_i(i) for i in range(len(weight))]
        loss = loss_function()
        print("Iteration %d weight %s, total loss: %f" % (step, str(weight), loss))

        # update weight
        for i in range(len(gradients)):
            weight[i] = weight[i] - learning_rate * gradients[i]

        if loss < early_stop:
            print("stop at iteration %d with loss function value %f" % (step, loss))
            break


train()
