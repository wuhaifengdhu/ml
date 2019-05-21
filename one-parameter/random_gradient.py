# First version of ML, h(x) = a * x + b
#
import random
raw_data = [(1.01, 0.99), (2.0, 2.01), (3.01, 2.999), (4.0005, 3.999)]

a = 0.5
b = 0
learning_rate = 0.01
training_iteration = 2100
early_stop = 0.0003


def gradient_a(x, y):
    global a, b
    return (a * x + b - y) * x


def gradient_b(x, y):
    global a, b
    return a * x + b - y


def loss_function():
    return 0.5 * sum([(a * x + b - y) * (a * x + b - y) for x, y in raw_data])


def train():
    global a,b, training_iteration, learning_rate, early_stop
    for i in range(training_iteration):
        record = random.choice(raw_data)
        a_gradient = gradient_a(* record)
        b_gradient = gradient_b(* record)
        loss = loss_function()
        print("Iteration %d function: y = %f * x + %f, total loss: %f, choose records: %s" % (i, a, b, loss, str(record)))
        a = a - learning_rate * a_gradient
        b = b - learning_rate * b_gradient
        if loss < early_stop:
            print("stop at iteration %d with loss function value %f" % (i, loss))
            break


train()

