# First version of ML, h(x) = a0 * x0 + a1 * x1 + a3 * x3
#
import numpy as np
import math
import random
raw_data = [(646.497831, 693.5535663, 463.4275959,181.5743217,622.1733223),(535.5969839,690.6967314,327.4093162,49.31105335,581.1116289),(668.0254323,681.2419271,419.2655195,139.2272431,610.9579079),(652.7649565,699.478907,426.6033777,186.2204538,622.3143211),(695.6679643,689.0674148,450.2171243,122.3169252,615.2969041),(538.3578654,687.6183486,327.5807353,51.87953382,580.8982459),(653.0445239,702.5127429,492.2533493,215.0790613,631.2851788),(646.497831,693.5535663,463.4275959,181.5743217,622.1733223)]


learning_rate = 0.0000001
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


def loss_function(record):
    return 0.5 * math.pow(compute_record_loss(record), 2)


def gradient_i(i, record):
    if i == 0:
        return compute_record_loss(record)
    else:
        return compute_record_loss(record) * record[i]


def get_mini_batch():
    mini_batch = [x for x in raw_data if random.random() > 0.3]
    return [random.choice(raw_data)] if len(mini_batch) == 0 else mini_batch


def train():
    for step in range(training_iteration):
        record = random.choice(raw_data)
        gradients = [gradient_i(i, record) for i in range(len(weight))]
        loss = loss_function(record)
        print("Iteration %d weight %s, total loss: %f" % (step, str(weight), loss))

        # update weight
        for i in range(len(gradients)):
            weight[i] = weight[i] - learning_rate * gradients[i]

        if loss < early_stop:
            print("stop at iteration %d with loss function value %f" % (step, loss))
            break


train()


