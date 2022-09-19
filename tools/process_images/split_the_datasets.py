import os  # os是用来切换路径和创建文件夹的。
import random
from shutil import copy  # shutil 是用来复制黏贴文件的

# "blossom_end_rot", "graymold", "powdery_mildew", "spider_mite", "spotting_disease"

img_path = "/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/JPEGImages"
labels_path = "/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/Annotations"

filelist_img_name = []
filelist_label_name = []

for root, dir, filenames in os.walk(img_path):
    for filename in filenames:
        if '.jpg' in filename:
            file_path = os.path.join(root, filename)
            print(file_path)
            filelist_img_name.append(file_path)
        else:
            pass
            # print("it is not a jpg file.", filename)

# .xml是我的标注文件的后缀名，您修改为自己的xml或者json或者其他类型即可。
for root, dir, filenames in os.walk(labels_path):
    for filename in filenames:
        if '.xml' in filename:
            filelist_label_name.append(filename.split('.xml')[0])
        else:
            pass
            # print("it is not a label file.", filename)

print(len(filelist_img_name))
print(len(filelist_label_name))
length = len(filelist_img_name)
a = int(length*0.8)
b = int(length*0.9)
# 我一共含有580张图片，因此 348 ，464，464-580这几个数字是我自己计算的。我的比例是60% 20% 20%
# 如果您有10000张图片和10000张标注，那么您可以设置为6000，8000，8000:，以此类推
# 我没有写自动创建文件夹的那部分代码，因此你要自己首先把相关文件夹创建好。
random.shuffle(filelist_img_name)
filelist_img_name = filelist_img_name

train_images_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/images/train/'
train_labels_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/labels/train/'
val_images_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/images/val/'
val_labels_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/labels/val/'
test_images_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/images/test/'
test_labels_path = '/Users/jiuqingdong/Documents/project_paper_code_data/yolov5-official_new/datasets/tomato_aug/labels/test/'
if not os.path.exists(train_images_path):
    os.mkdir(train_images_path)
if not os.path.exists(train_labels_path):
    os.mkdir(train_labels_path)
if not os.path.exists(val_images_path):
    os.mkdir(val_images_path)
if not os.path.exists(val_labels_path):
    os.mkdir(val_labels_path)
if not os.path.exists(test_images_path):
    os.mkdir(test_images_path)
if not os.path.exists(test_labels_path):
    os.mkdir(test_labels_path)

for tr in filelist_img_name[0:a]:
    copy(tr, train_images_path)
    print(tr)
    tr = tr.replace('JPEGImages/', 'Annotations/')
    tr = tr.replace('.jpg', '.xml')
    copy(tr, train_labels_path)

for val in filelist_img_name[a:b]:
    copy(val, val_images_path)
    val = val.replace('JPEGImages/', 'Annotations/')
    val = val.replace('.jpg', '.xml')
    copy(val, val_labels_path)

for test in filelist_img_name[b:]:
    copy(test, test_images_path)
    test = test.replace('JPEGImages/', 'Annotations/')
    test = test.replace('.jpg', '.xml')
    copy(test, test_labels_path)

'''
# matching the images to labels

for k in filelist_img_name:
    if k not in filelist_json_name:
    #print("there is no label of image {}".format(k))
        pass
    else:
        print("ok")

'''
