import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.io as sio

# for graphing 
def graph(function):
	x = np.array(range(0,80))  
	y = eval(function)
	plt.plot(x, y) 


Il = sio.loadmat('edges.mat')
I = Il['bw05']
# print(I)

# plt.matshow(I)
plt.imshow(I, cmap='Greys')


graph('-3.75/0.45 + (0.88/0.45)*x')
graph('6/0.91 + (0.41/0.91)*x')
graph('87/0.61 - (0.73/0.61)*x')

plt.show()