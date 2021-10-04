import matplotlib.pyplot as plt
import numpy as np

pi = np.pi
a = np.linspace(0, 2*pi, 1000)
b = np.sin(a)
plt.plot(a, b, c='b', label='y = sin(x)')
plt.legend()
plt.title('Sin Function')
plt.show()

