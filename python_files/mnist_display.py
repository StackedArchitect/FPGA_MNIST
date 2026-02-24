import numpy as np
import matplotlib.pyplot as plt

with open("data_in.mem",'r') as file:
    data = file.readlines()


data = [int(i.strip(),16) for i in data]

mnist_data = data[20:-20]
mnist_8bit = np.array(mnist_data, dtype=np.uint8).reshape(28, 28)

plt.imshow(mnist_8bit, cmap = 'gray')
plt.show()