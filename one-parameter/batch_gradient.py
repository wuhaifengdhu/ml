# First version of ML, h(x) = a * x + b
#
raw_data = [(1.01, 0.99), (2.0, 2.01), (3.01, 2.999), (4.0005, 3.999)]

a = 0.5
b = 0.5
learning_rate = 0.01
training_iteration = 210000
early_stop = 0.0003


def gradient_a():
    global a, b, raw_data
    return sum((y - (a * x + b)) * x for x, y in raw_data)


def gradient_b():
    global a, b, raw_data
    return sum(y - (a * x + b) for x, y in raw_data)


def loss_function():
    return 0.5 * sum([(a * x + b - y) * (a * x + b - y) for x, y in raw_data])


def train():
    global a,b, training_iteration, learning_rate, early_stop
    for i in range(training_iteration):
        a_gradient = gradient_a()
        b_gradient = gradient_b()
        loss = loss_function()
        print("Iteration %d function: y = %f * x + %f, total loss: %f" % (i, a, b, loss))
        a = a + learning_rate * a_gradient
        b = b + learning_rate * b_gradient
        if loss < early_stop:
            print("stop at iteration %d with loss function value %f" % (i, loss))
            break

train()
