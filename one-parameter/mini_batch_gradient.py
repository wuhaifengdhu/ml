import random

raw_data = [(1.01, 0.99), (2.0, 2.01), (3.01, 2.999), (4.0005, 3.999)]

a = 0.5
b = 0.5
learning_rate = 0.01
training_iteration = 21000
early_stop = 0.0003


def get_mini_batch():
    global raw_data
    mini_batch = [x for x in raw_data if random.random() > 0.3]
    return [random.choice(raw_data)] if len(mini_batch) == 0 else mini_batch


def gradient_a(mini_batch):
    global a, b
    return sum((y - (a * x + b)) * x for x, y in mini_batch)


def gradient_b(mini_batch):
    global a, b
    return sum(y - (a * x + b) for x, y in mini_batch)


def loss_function():
    return 0.5 * sum([(a * x + b - y) * (a * x + b - y) for x, y in raw_data])


def train():
    global a,b, training_iteration, learning_rate, early_stop
    for i in range(training_iteration):
        mini_batch = get_mini_batch()
        a_gradient = gradient_a(mini_batch)
        b_gradient = gradient_b(mini_batch)
        loss = loss_function()
        print("Iteration %d function: y = %f * x + %f, total loss: %f, choose data: %s" % (i, a, b, loss, str(mini_batch)))
        a = a + learning_rate * a_gradient
        b = b + learning_rate * b_gradient
        if loss < early_stop:
            print("stop at iteration %d with loss function value %f" % (i, loss))
            break


train()
print("compare with base loss %f" % (0.5 * sum([(x - y) * (x - y) for x, y in raw_data])))

