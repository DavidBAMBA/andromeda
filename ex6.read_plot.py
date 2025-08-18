import numpy as np
import matplotlib.pyplot as plt

filename = 'Kerr_a_0.9_1920x1080_No_Doppler'

image_data = np.load('images_data/' + filename + '.npy')
image_data = image_data/image_data.max()
ax = plt.figure().add_subplot(aspect='equal')
ax.contour(image_data.T, cmap='gray')
plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'$\beta$', rotation=0)
ax.grid(alpha=0.25)
plt.savefig('images/Contours_'+filename+'.png')
plt.show()