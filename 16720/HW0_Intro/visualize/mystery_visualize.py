#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap. See the Bewares
    """
    H, W, C = X.shape
    N, _ = colors.shape # N is num of colors
    vmin = np.nanmin(X)
    vmax = np.nanmax(X)
    if vmin == vmax: # Edge case to avoid division by zero in color equation
        vmax = vmin + 1

    O_list = []

    for c in range(C):
        # O = np.zeros((H, W, 3), dtype=np.uint8)
        # for h in range(H):
        #     for w in range(W):
        #         value = X[h,w,c]
        #         if  not np.isfinite(value):
        #             continue
        #         # Use the color equation to get the color
        #         color_index = int((N - 1) * ((value - vmin) / (vmax - vmin)))
        #         color_index = max(0, min(color_index, N - 1)) # Clamp to 0-N-1
        #         O[h,w, :] = (colors[color_index] * 255).astype(np.uint8) # Scale to 0-255 from 0-1

        O = X[:,:,c]
        color_index = (N-1)*((O - vmin) / (vmax - vmin))
        color_index = np.clip(color_index, 0, N-1).astype(np.uint8)
        O = colors[color_index]
        O = (O * 255).astype(np.uint8) # Scale to 0-255 from 0-1
        
        plt.imsave("exports/mysterydata4/colormapArray_%d.png" % c, O)

        O_list.append(O)

    return O_list


if __name__ == "__main__":
    ## mysterydata.npy #########################################################
    os.makedirs("exports/mysterydata", exist_ok=True)
    data = np.load("mysterydata/mysterydata.npy")

    print("data", data, data.shape, data.dtype)

    # False color image of the first channel
    channel = 0
    plt.imsave("exports/mysterydata/vis.png",data[:,:,channel])

    # Looking at all the channels
    for i in range(data.shape[2]):
        plt.imsave("exports/mysterydata/vis_%d.png" % i,data[:,:,i], cmap='plasma')


    ## mysterydata2.npy #########################################################
    os.makedirs("exports/mysterydata2", exist_ok=True)

    data2 = np.load("mysterydata/mysterydata2.npy")
    print("data2", data2, data2.shape, data2.dtype)

    for i in range(data2.shape[2]):
        plt.imsave("exports/mysterydata2/vis2_%d.png" % i,data2[:,:,i])

    data2_pow = np.power(data2, 0.3)
    for i in range(data2.shape[2]):
        plt.imsave("exports/mysterydata2/vis2_pow_%d.png" % i,data2_pow[:,:,i])

    epsilon = 1e-6 # Avoid log(0)
    data2_log = np.log(data2 + epsilon)
    for i in range(data2.shape[2]):
        plt.imsave("exports/mysterydata2/vis2_log_%d.png" % i,data2_log[:,:,i])

    data2_log1p = np.log1p(data2)
    for i in range(data2.shape[2]):
        plt.imsave("exports/mysterydata2/vis2_log1p_%d.png" % i,data2_log1p[:,:,i])


    ## mysterydata3.npy #########################################################
    os.makedirs("exports/mysterydata3", exist_ok=True)
    data3 = np.load("mysterydata/mysterydata3.npy")
    for i in range(9):
        plt.imsave("exports/mysterydata3/vis3_%d.png" % i,data3[:,:,i])
    # Images are blank, check for NaN or inf
    print("Should be 1: ", np.mean(np.isfinite(data3)))
    print("Number not finite: ", np.sum(~np.isfinite(data3)))

    # Find min and max, ignoring NaN
    finite_min = np.nanmin(data3)
    finite_max = np.nanmax(data3)
    print("Min: ", finite_min, "Max: ", finite_max)
    for i in range(9):
        plt.imsave("exports/mysterydata3/vis3_fixed_%d.png" % i,data3[:,:,i], vmin=finite_min, vmax=finite_max)

    ## mysterydata4.npy #########################################################
    os.makedirs("exports/mysterydata4", exist_ok=True)
    colors = np.load("mysterydata/colors.npy")
    data4 = np.load("mysterydata/mysterydata4.npy")

    colormapArray(data4, colors)

    print("colors", colors, colors.shape, colors.dtype)
    print("data4", data4, data4.shape, data4.dtype)
    for i in range(data4.shape[2]):
        plt.imsave("exports/mysterydata4/vis4_%d.png" % i,data4[:,:,i])

    # pdb.set_trace()
