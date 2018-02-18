"""
    PA2 - Canny Edge Detector. This program is designed to create a canny style edge detector. Using the images from PA1
    the program will first implement a gaussian filter, take that image and use a gradient function to obtain gradient
    images, Implement a non-max suppression algorithm, use hystersis thresholding and display the final canny edge
    result.
"""
import sys
import cv2
import numpy as np
import scipy.stats as st
import scipy.misc as im


def thresholding(image, high, low, orig_image):

    strongEdges = (image > int(high))
    thresholdedEdges = np.array(strongEdges, dtype=np.uint8) + (image > int(low))

    finalEdges = strongEdges.copy()
    currentPixels = []
    for r in range(1, orig_image.shape[0] - 1):
        for c in range(1, orig_image.shape[1] - 1):
            if thresholdedEdges[r, c] != 1:
                continue  # Not a weak pixel

            # Get 3x3 patch
            localPatch = thresholdedEdges[r - 1:r + 2, c - 1:c + 2]
            patchMax = localPatch.max()
            if patchMax == 2:
                currentPixels.append((r, c))
                finalEdges[r, c] = 1

    # Extend strong edges based on current pixels
    while len(currentPixels) > 0:
        newPix = []
        for r, c in currentPixels:
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0: continue
                    r2 = r + dr
                    c2 = c + dc
                    if thresholdedEdges[r2, c2] == 1 and finalEdges[r2, c2] == 0:
                        # Copy this weak pixel to final result
                        newPix.append((r2, c2))
                        finalEdges[r2, c2] = 1
        currentPixels = newPix

    print(finalEdges)

    return finalEdges


def supression(image, image_orig):
    """
    non-maximum supression algorithm

    :param image_orig:
    :param image: class: tuple (gradient magnitude ndarray, theta)
    :return: new nd suppressed array.
    """
    gradSup = image[0].copy()
    thetaQ = image[1].copy()
    grad_mag = image[0].copy()

    for i in range(image_orig.shape[0]):
        for j in range(image_orig.shape[1]):
            # Suppress pixels at the image edge
            if i == 0 or i == image_orig.shape[0] - 1 or j == 0 or j == image_orig.shape[1] - 1:
                gradSup[i, j] = 0
                continue
            tq = thetaQ[i, j] % 4
            if tq == 0:  # 0 is E-W (horizontal)
                if grad_mag[i, j] <= grad_mag[i, j - 1] or grad_mag[i, j] <= grad_mag[i, j + 1]:
                    gradSup[i, j] = 0
            if tq == 1:  # 1 is NE-SW
                if grad_mag[i, j] <= grad_mag[i - 1, j + 1] or grad_mag[i, j] <= grad_mag[i + 1, j - 1]:
                    gradSup[i, j] = 0
            if tq == 2:  # 2 is N-S (vertical)
                if grad_mag[i, j] <= grad_mag[i - 1, j] or grad_mag[i, j] <= grad_mag[i + 1, j]:
                    gradSup[i, j] = 0
            if tq == 3:  # 3 is NW-SE
                if grad_mag[i, j] <= grad_mag[i - 1, j - 1] or grad_mag[i, j] <= grad_mag[i + 1, j + 1]:
                    gradSup[i, j] = 0
    return gradSup


def gradient(image_gaussian, image_original):
    """
    gradient function - takes an image and creates two gradient images (x dir and y dir) along with the gradient
    magnitude of that image. returns a list of the gradient images and magnitude images.

    :param image_original:
    :param image_gaussian: gaussian manipulated image (ndarray)
    :return: gradient representations of the image along with the gradient magnitude.
    """
    grad_image = np.gradient(image_gaussian * image_original)
    grad_mag = np.power(np.power(grad_image[0], 2) + np.power(grad_image[1], 2), .5)
    grad_ori = np.arctan2(grad_image[0], grad_image[1])
    thetaQ = (np.round(grad_ori * (5.0 / np.pi)) + 5) % 5

    image = [grad_mag, thetaQ]

    return image


def gaussian(kernel_size, sigma, image):
    """
    gaussian filter - implements a 2d gaussian filter and iterates over the image using the built in convolution
    """

    interval = (2 * int(sigma) + 1.) / (int(kernel_size))
    x = np.linspace(-int(sigma) - interval / 2., int(sigma) + interval / 2., int(kernel_size) + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel.reshape(int(kernel_size), int(kernel_size))

    processed_image = cv2.filter2D(image, -1, kernel)

    return processed_image


def main(image, high_t, low_t):
    """
    main function - takes the argument functions and creates the procedural timeline for the program.

    :param image: String representation of the image path.
    :param sigma: value of sigma for the gaussian filter.
    :param kernel: value for the kernel size for the gaussian filter.
    :return: ndarray
    """
    read_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    gaussian_image = gaussian(13, 2, read_image)
    grad_image = gradient(gaussian_image, read_image)
    supressed_image = supression(grad_image, read_image)
    bin_image = thresholding(supressed_image, high_t, low_t, read_image)
    im.imshow(bin_image)
    cv2.waitKey(0)

    return 0


res = main(sys.argv[1], sys.argv[2], sys.argv[3])

# EOF
