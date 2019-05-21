# First version of ML, h(x) = a * x + b
#
raw_data = (1.01, 0.99)

a = 0.5
b = 0.5
learning_rate = 0.01
training_iteration = 2100


def gradient_a():
    global a, b, raw_data
    return (a * 1.01 + b - 0.99) * 1.01


def gradient_b():
    global a, b, raw_data
    return a * 1.01 + b - 0.99


def train():
    global a,b, training_iteration, learning_rate
    for i in range(training_iteration):
        a_gradient = gradient_a()
        b_gradient = gradient_b()
        loss = gradient_b()
        print("Iteration %d function: y = %f * x + %f, total loss: %f" % (i, a, b, loss))
        a = a - learning_rate * a_gradient
        b = b - learning_rate * b_gradient


train()
