from engine import Value
from nn import Neuron, Layer, MLP


# testing on a small [3, 4, 4, 1] MLP
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
n(x) # initialize the MLP

# make some synthetic data
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]



# training
for k in range(1001):

  # forward pass
  ypred = [n(x) for x in xs]
  loss = sum(((yout - ygt)**2 for ygt, yout in zip(ys, ypred)))
  
  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update with GD
  for p in n.parameters():
    p.data -= 0.05 * p.grad

  if k % 100 == 0:
      print(f"epoch: {k}, loss: {loss.data:.4f}")


# predictions
for i in range(len(ypred)):
   print(ypred[i])
