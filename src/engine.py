# Value object

import math

class Value:
  def __init__(self, data, _children=(), _op='', label=""):
    self.data = data
    self.grad = 0.0
    self._backward = lambda: None
    self._prev = set(_children)
    self._op = _op
    self.label = label


  def __repr__(self):
    return f"Value(data={self.data})"


  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other) # so constants can be added to Value instances as well
    sum = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.grad += 1.0 * sum.grad
      other.grad += 1.0 * sum.grad
    sum._backward = _backward

    return sum

  # fallback in case of constant + Value
  def __radd__(self, other): # basically: other * self
    return self + other


  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    prod = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.grad += other.data * prod.grad
      other.grad += self.data * prod.grad

    prod._backward = _backward

    return prod


  # fallback function when doing something like const * Value
  # Python will check if Value can multiply const. if yes, it does rmul
  # instead of mul
  def __rmul__(self, other): # basically: other * self
    return self * other


  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self, ), f'**{other}')

    def _backward():
      self.grad = other * self.data**(other-1) * out.grad
    out._backward = _backward

    return out


  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1


  def tanh(self):
    x = self.data
    t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
    out = Value(t, (self, ), 'tanh')

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out

    def _backward():
      self.grad += (1 - t**2) * out.grad
    out._backward = _backward

    return out


  def sigmoid(self):
    # f(x) = 1/(1 + e^(-x))
    x = self.data 

    if x >= 0 :
      sig = 1/(1 + math.exp(-x))
    else:
      sig = math.exp(x) / (1 + math.exp(x))
    out = Value(sig, (self, ), 'sigmoid')

    def _backward():
      self.grad += (sig*(1 - sig)) * out.grad
    out._backward = _backward

    return out


  def relu(self):
    """ReLU activation function: max(0, x)"""
    x = self.data
    out = Value(max(0, x), (self, ), 'relu')
    
    def _backward():
        self.grad += (x > 0) * out.grad
    out._backward = _backward
    
    return out

  def leaky_relu(self, alpha=0.01):
    # f(x) = max(alpha*x, x)
    x = self.data
    out = Value(x if x > 0 else alpha*x, (self, ), f'leaky_relu{alpha}')

    def _backward():
      self.grad += (1 if x > 0 else alpha) * out.grad
    out._backward = _backward

    return out


  def softplus(self):
    # f(x) = log(1 + exp(x))
    x = self.data
    # for nuemrical stability, setting a threshold
    if x > 20:
      result = x
    else:
      result = math.log(1 + math.exp(x))
    out = Value(result, (self, ), 'softplus')

    def _backward():
        # Derivative is sigmoid
        if x > 20:  # For numerical stability
            sigmoid_x = 1.0
        else:
            sigmoid_x = 1 / (1 + math.exp(-x))
        self.grad += sigmoid_x * out.grad
    out._backward = _backward

    return out


  # ----------------------------------


  def exp(self):
    x = self.data
    exp = Value(math.exp(x), (self, ), 'exp')

    def _backward():
      self.grad += exp.data * exp.grad # chain rule
    exp._backward = _backward

    return exp


  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)


  def backward(self):

    topo = []
    visited = set()
    def build_topo(v):
      if v not in visited:
        visited.add(v)
        for child in v._prev:
          build_topo(child)
        topo.append(v)
    build_topo(self)

    self.grad = 1.0
    for node in reversed(topo):
      node._backward()