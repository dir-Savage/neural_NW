import math
import random

def exp_fn(x, terms=30):
    res, term = 1.0, 1.0
    for i in range(1, terms):
        term *= x / i
        res += term
    return res

def tanh_fn(x):
    e = exp_fn(2 * x)
    return (e - 1) / (e + 1)

def rand_val(low, high):
    return random.uniform(low, high)

def init_weights(structure):
    return [[rand_val(-0.5, 0.5) for _ in range(structure[i] * structure[i + 1])] for i in range(len(structure) - 1)]

def init_biases(structure):
    return [[rand_val(-0.5, 0.5) for _ in range(structure[i])] for i in range(1, len(structure))]

def fwd_pass(inputs, weights, biases):
    activations = inputs
    for w, b in zip(weights, biases):
        next_act = []
        for j in range(len(b)):
            act = sum(a * w[i + j * len(activations)] for i, a in enumerate(activations)) + b[j]
            next_act.append(tanh_fn(act))
        activations = next_act
    return activations

def build_nn(in_dim, hidden, out_dim):
    struct = [in_dim] + hidden + [out_dim]
    weights = init_weights(struct)
    biases = init_biases(struct)
    return weights, biases

in_dim = 2
hidden_layers = [6, 4, 3]
out_dim = 2
weights, biases = build_nn(in_dim, hidden_layers, out_dim)
inputs = [0.05, 0.10]
outputs = fwd_pass(inputs, weights, biases)
print(f"NN Output: {', '.join(f'{o:.4f}' for o in outputs)}")
