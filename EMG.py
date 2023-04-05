##
# Annie Bonde
# File for running the EM class given the filename and k and flag
# flag = 0 = unimproved model, fla=1: improved model

from EM import EM
import skimage
import matplotlib.pyplot as plt
import numpy as np

def EMG(file_name, k, flag=0):
    image = skimage.io.imread(file_name)
    image = image / 255.
    # flatten image so there are three cols (rbg)
    data_df = np.transpose(np.array([image[:, :, 0].flatten(), image[:, :, 1].flatten(), image[:, :, 2].flatten()]))

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title("Original Image")
    model = EM(data_df, k)
    if flag == 0:
        model.fit()
    if flag == 1:
        model.fit2()
    model.predict(ax2, image.shape)
    plt.show()

    fig2, ax3 = plt.subplots()
    model.plot_logs(ax3)

    plt.show()

