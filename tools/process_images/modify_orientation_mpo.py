import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

# image_dir = '/Users/jiuqingdong/Desktop/test/JPEGImages/'
image_dir = '/Volumes/Data/project_paper_code_data/Plant_diseases_dataset/Tomato_dataset/JPEGImages/'
output_dir = '/Volumes/Data/project_paper_code_data/Plant_diseases_dataset/Tomato_dataset/JPEGImages_modify_orientation/'


def exif_transpose_hand(img, orientation):
    #if not img:
    #    return img
    if orientation == None:

        return img
    else:
        image = img.rotate(0, expand=True)

    return image


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274
    image = img
    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]
        print("ori",orientation)
        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            image = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            image = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            image = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            image = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            image = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            image = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            image = img.rotate(90, expand=True)
        elif orientation == None:
            image = img.rotate(0, expand=True)
    return image


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)
    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        # print(" if hasattr(PIL.ImageOps, 'exif_transpose'):")
        a = img.getexif().get(0x0112)
        # if a == 1  or a == None:
        print(file, ' = = = = = ', a)

        #img = PIL.ImageOps.exif_transpose(img)

        image = exif_transpose_hand(img, a)
    else:
        print("=else=")
        # Otherwise, do the exif transpose ourselves
        # image = exif_transpose(img)

    image = image.convert(mode)
    #return img
    return np.array(image)

# image_dir = '/Users/Dong/Desktop/project_paper_code_data/Plant_diseases_dataset/Paprika Dataset_v_1.02/images'

# for set in ['train/', 'val/', 'test/']:
#     image_dir = image_dir0+set
#     output_dir = output_dir0+set

for root, dirs, files in os.walk(image_dir):
    for file in files:
        filename = os.path.join(root, file)
        img = load_image_file(filename)

        # plt.imshow(img)
        plt.imsave(output_dir+file, img)

        #plt.pause(1)
        #plt.close()
