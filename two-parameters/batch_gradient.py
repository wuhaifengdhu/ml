# First version of ML, h(x) = a0 * x0 + a1 * x1 + a3 * x3
#
import numpy as np
import math
import random
raw_data = [(646.497831, 693.5535663, 463.4275959,181.5743217,622.1733223),
            (535.5969839,690.6967314,327.4093162, 49.31105335,581.1116289),
            (668.0254323,681.2419271,419.2655195,139.2272431,610.9579079),
            (652.7649565,699.478907,426.6033777, 186.2204538,622.3143211),
            (695.6679643,689.0674148,450.2171243,122.3169252,615.2969041),
            (538.3578654,687.6183486,327.5807353,51.87953382,580.8982459),
            (653.0445239,702.5127429,492.2533493,215.0790613,631.2851788),
            (646.497831,693.5535663,463.4275959,181.5743217, 622.1733223)]

eval_data = [(48.3861111, 529.934835, 25.398319, 9.87610134, 383.618488),
             (226.234598, 565.13408, 173.943412, 94.2828818, 482.751501)]

learning_rate = 0.00000008
training_iteration = 17

weight = [random.random() for ite in range(len(raw_data[0]))]
# weight = np.zeros(len(raw_data[0]))
# weight = np.ones(len(raw_data[0]))
gradient = []


def compute_record_loss(record):
    record_loss = forward(record) - record[-1]
    return record_loss


def loss_function():
    total_loss = sum([math.pow(compute_record_loss(record), 2) for record in raw_data])
    return total_loss * 0.5


def forward(record):
    predict = weight[0] * 1  # calculate the first weight is present
    for i in range(len(record) - 1):  # the last record is target, so minus 1 here
        predict = predict + weight[i + 1] * record[i]
    return predict


def gradient_i(it):
    if it == 0:
        return sum([compute_record_loss(record) for record in raw_data])
    else:
        return sum([compute_record_loss(record) * record[it] for record in raw_data])


loss_list = []


def early_stop_check():
    if len(loss_list) < 2:
        return False
    return loss_list[-1] >= loss_list[-2]


def train():
    for step in range(training_iteration):
        gradients = [gradient_i(i) for i in range(len(weight))]
        loss = loss_function()
        print("Iteration %d weight %s, total loss: %f" % (step, str(weight), loss))

        # update weight
        for i in range(len(gradients)):
            weight[i] = weight[i] - learning_rate * gradients[i]

        loss_list.append(loss)
        if early_stop_check():
            print("stop at iteration %d with loss function value %f" % (step, loss))
            break


def eval_model():
    for data in eval_data:
        predict = forward(data)
        print("predict value is %f, actually is %f, loss is %s" % (predict, data[-1], (predict - data[-1])))


train()
eval_model()

# eval_data = [(1, 0, 0, 0, 1),
#              (0, 1, 0, 0, 1)]
#
# def eval_model():
#     for data in eval_data:
#         predict = forward(data)
#         print("predict value is %f, actually is %f, loss is %s" % (predict, data[-1], (predict - data[-1])))
