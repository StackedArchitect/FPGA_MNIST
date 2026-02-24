from scipy.special import softmax
import numpy as np


def to_float(x,e):
    c = abs(x)
    sign = 1 
    if x < 0:
        c = x - 1 
        c = ~c
        sign = -1
    f = (1.0 * c) / (2 ** e)
    f = f * sign
    return f


fixed_values = [  -238614955366566,
      -44428701398369,
     -166458277074892,
      152071442292679,
     -352221585710395,
     -136088124260311,
     -570904630356912,
      616370329258801,
     -140338139838807,
      286630486585060,]
floating_values = []

for i in fixed_values:
    n = to_float(i,16)
    floating_values.append(n)

print(f'floating point values are: {floating_values}')


result = softmax(floating_values)
print(f'softmax values {result}')
print(f'Predicted Digit: {np.argmax(result)}')


