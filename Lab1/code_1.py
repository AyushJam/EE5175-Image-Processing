"""
Code for Lab-I: EE5175

@author: Ayush Jamdar EE20B018
"""

import numpy as np
from PIL import Image


def image2arr(filename):
    image = Image.open(filename)
    return np.array(image)


def saveArrAsImage(image_array, filepath, file_format):
    image = Image.fromarray(image_array)
    # image.show()
    image.save(filepath, file_format)


def translate(tx, ty, source_filename, dest_filepath, file_format):
    im_arr = image2arr(source_filename)

    translated_image = np.zeros(im_arr.shape).astype(np.uint8)

    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            xs = i - tx
            ys = j - ty
            xs_flr = int(np.floor(xs))
            ys_flr = int(np.floor(ys))

            if xs_flr >= 0 and ys_flr >= 0:
                a = xs - xs_flr
                b = ys - ys_flr
                translated_image[i, j] = (
                    (1 - a) * (1 - b) * im_arr[xs_flr, ys_flr]
                    + (1 - a) * b * im_arr[xs_flr, ys_flr + 1]
                    + a * (1 - b) * im_arr[xs_flr + 1, ys_flr]
                    + a * b * im_arr[xs_flr + 1, ys_flr + 1]
                )

    saveArrAsImage(translated_image, dest_filepath, file_format)
    return


def rotate(source_filename, dest_filepath, file_format):
    image_array = image2arr(source_filename)
    theta = -5  # degrees

    # rotation matrix
    R = np.array(
        [
            [np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta))],
            [-np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))],
        ]
    )

    rotated_image = np.zeros(image_array.shape).astype(np.uint8)

    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            # shift origin from corner to center of image
            xs_shifted, ys_shifted = np.matmul(
                np.linalg.inv(R),
                np.array((i - image_array.shape[0] / 2, j - image_array.shape[1] / 2)),
            )

            # shift origin back to image corner
            xs = xs_shifted + image_array.shape[0] / 2
            ys = ys_shifted + image_array.shape[1] / 2

            if (0 <= xs < image_array.shape[0] - 1) and (
                0 <= ys < (image_array.shape[1] - 1)
            ):
                xs_flr = int(np.floor(xs))
                ys_flr = int(np.floor(ys))

                a = xs - xs_flr
                b = ys - ys_flr
                rotated_image[i, j] = (
                    (1 - a) * (1 - b) * image_array[xs_flr, ys_flr]
                    + (1 - a) * b * image_array[xs_flr, ys_flr + 1]
                    + a * (1 - b) * image_array[xs_flr + 1, ys_flr]
                    + a * b * image_array[xs_flr + 1, ys_flr + 1]
                )

    # save as image
    saveArrAsImage(rotated_image, dest_filepath, file_format)

    return


def scale(xscaling_factor, yscaling_factor, source_filename, dest_filepath, file_format):
    im_arr = image2arr(source_filename)

    scaled_image = np.zeros(im_arr.shape).astype(np.uint8)

    for i in range(im_arr.shape[0]):
        for j in range(im_arr.shape[1]):
            xs = i / xscaling_factor
            ys = j / yscaling_factor
            xs_flr = int(np.floor(xs))
            ys_flr = int(np.floor(ys))

            if xs_flr < im_arr.shape[0] and ys_flr < im_arr.shape[1]:
                a = xs - xs_flr
                b = ys - ys_flr
                scaled_image[i, j] = (
                    (1 - a) * (1 - b) * im_arr[xs_flr, ys_flr]
                    + (1 - a) * b * im_arr[xs_flr, ys_flr + 1]
                    + a * (1 - b) * im_arr[xs_flr + 1, ys_flr]
                    + a * b * im_arr[xs_flr + 1, ys_flr + 1]
                )

    saveArrAsImage(scaled_image, dest_filepath, file_format)
    return


def assignment():
    translate(
        tx=3.75,
        ty=4.3,
        source_filename="lena_translate.png",
        dest_filepath="lena_translated.png",
        file_format="PNG",
    )

    rotate("pisa_rotate.png", "pisa_rotated.png", "PNG")

    scale(
        xscaling_factor=0.8,
        yscaling_factor=1.3,
        source_filename="cells_scale.png",
        dest_filepath="scaled_xy.png",
        file_format="PNG",
    )

    return


assignment()
