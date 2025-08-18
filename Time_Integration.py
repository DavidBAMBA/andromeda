import numpy as np
import matplotlib.pyplot as plt


data = np.array([[64**2, 128**2, 256**2, 512**2, 1024**2, 2048**2], 
                 [11.419, 47.195, 185.793, 730.904, 2902.645, 12564.2]])

plt.figure()
plt.plot(data[0], data[1])
plt.show()