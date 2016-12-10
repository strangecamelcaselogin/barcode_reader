import matplotlib.pyplot as plt
import numpy as np

plt.ion()
a = plt.figure()

for count in range(14):
    a.clear()
    x = np.arange(-10+count, 10+count, 0.1)
    y = np.sin(x)
    plt.plot(x, y)
    plt.draw()

    plt.pause(0.0000000000001)
    print(count)


plt.close(a)
