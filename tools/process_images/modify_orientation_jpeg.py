import PIL.Image
import PIL.ImageOps
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def load_image_file(file, mode='RGB'):
    # Load the image with PIL
    img = PIL.Image.open(file)
    if hasattr(PIL.ImageOps, 'exif_transpose'):
        # Very recent versions of PIL can do exit transpose internally
        # print(" if hasattr(PIL.ImageOps, 'exif_transpose'):")
        a = img.getexif().get(0x0112)
        # if a == 1  or a == None:
        print(file, ' = = = = = ', a)
        img = PIL.ImageOps.exif_transpose(img)

        # img = exif_transpose(img)
    else:
        print("=else=")
        # Otherwise, do the exif transpose ourselves
        img = exif_transpose(img)

    img = img.convert(mode)
    #return img
    return np.array(img)

# image_dir = '/Users/Dong/Desktop/project_paper_code_data/Plant_diseases_dataset/Paprika Dataset_v_1.02/images'
image_dir = '/Users/jiuqingdong/Desktop/test/JPEGImages/'
output_dir = '/Users/jiuqingdong/Desktop/test/train/'

# for set in ['train/', 'val/', 'test/']:
#     image_dir = image_dir0+set
#     output_dir = output_dir0+set

for root, dirs, files in os.walk(image_dir):
    for file in files:
        filename = os.path.join(root, file)
        img = load_image_file(filename)

        plt.imshow(img)
        plt.imsave(output_dir+file, img)
        plt.pause(1)
        plt.close()
