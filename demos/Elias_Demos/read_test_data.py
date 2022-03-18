import numpy as np
from phi.flow import *

a = np.load("test.npz")
data = a['test']

import matplotlib.pyplot as plt

# plot lines
for i, lines in enumerate(data):
    plt.plot(lines[0], lines[1], label=f"line {i}")

plt.legend()
plt.show()

print("done")
