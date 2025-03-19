from engine import Value
import random

class Neuron:

  def __init__(self, n_in): # num of inputs
    # initialize the weights and bias
    self.w = [Value(random.uniform(-1,1)) for _ in range(n_in)]
    self.b = Value(random.uniform(-1,1))

  def __call__(self, x):
    # w*x + b
    activation = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    output = activation.tanh()
    return output

  def parameters(self):
    return self.w + [self.b]


class Layer:

  def __init__(self, n_in, n_out):
    self.neurons = [Neuron(n_in) for _ in range(n_out)]

  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs)==1 else outs

  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

  def __init__(self, n_in, n_outs):
    size = [n_in] + n_outs
    self.layers = [Layer(size[i], size[i+1]) for i in range(len(n_outs))]

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x

  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

