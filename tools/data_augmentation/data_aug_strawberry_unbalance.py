import random
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import shutil
import imgaug as ia
from imgaug import augmenters as iaa
from tqdm import tqdm
ia.seed(1)


SAVE = True


def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    # bndbox = root.find('object').find('bndbox')
    return bndboxlist


# (506.0000, 330.0000, 528.0000, 348.0000) -> (520.4747, 381.5080, 540.5596, 398.6603)
def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str("%06s" % (str(id) + '.xml'))))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 这里root分别由两个意思
    tree = ET.parse(in_file)
    elem = tree.find('filename')
    id = str(id)
    elem.text = str("%06s" % id) + '.jpg'
    xmlroot = tree.getroot()
    index = 0

    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        xmin_org = int(bndbox.find('xmin').text)
        xmax_org = int(bndbox.find('xmax').text)
        ymin_org = int(bndbox.find('ymin').text)
        ymax_org = int(bndbox.find('ymax').text)

        area_org = int((xmax_org - xmin_org) * (ymax_org - ymin_org))

        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        area_new = int((new_xmax - new_xmin) * (new_ymax -new_ymin))

        if new_ymax <= new_ymin or new_xmax <= new_xmin:
            xmlroot.remove(object)
        elif abs(area_new) <= abs(area_org/2):
            xmlroot.remove(object)
        else:
            xmin = bndbox.find('xmin')
            xmin.text = str(new_xmin)
            ymin = bndbox.find('ymin')
            ymin.text = str(new_ymin)
            xmax = bndbox.find('xmax')
            xmax.text = str(new_xmax)
            ymax = bndbox.find('ymax')
            ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str("%06s" % id) + '.xml'))


def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":
    # 原始的img与xml文件路径
    set='val'
    IMG_DIR = "/Users/Dong/Documents/test_strawberry/dataset_org/"+set
    XML_DIR = "/Users/Dong/Documents/test_strawberry/dataset_org/"+set+"_xml"
    # 存储增强后的XML文件夹路径
    AUG_XML_DIR = "/Users/Dong/Documents/test_strawberry/dataset_aug/"+set+"_xml"
    try:
        shutil.rmtree(AUG_XML_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_XML_DIR)
    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = "/Users/Dong/Documents/test_strawberry/dataset_aug/"+set
    try:
        shutil.rmtree(AUG_IMG_DIR)
    except FileNotFoundError as e:
        a = 1
    mkdir(AUG_IMG_DIR)
    # 每张影像增强的数量
    AUGLOOP = 0

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # 对50%的图像做镜像翻转
        #iaa.Rotate(-5,5),
        # iaa.ContrastNormalization((0.75, 1.5), per_channel=True),
        iaa.Crop(percent=(0, 0.05), keep_size=True),
        # iaa.Multiply((1.2, 1.5)),  # 改变亮度
        iaa.GaussianBlur(sigma=(0, 0.5)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            rotate=(-90, -90)
        )  # 对一部分图像做仿射变换, rotate旋转±30度之间, scale图像缩放为80%到95%之间, translate_px 独立地在x轴和y轴上将图像平移到15像素
    ])
    seq2 = iaa.Sequential([
        iaa.Flipud(0.5),  # 对50%的图像做镜像翻转
        # iaa.ContrastNormalization((0.75, 1.5), per_channel=True),
        iaa.Crop(percent=(0, 0.05), keep_size=True),
        # iaa.Multiply((1.2, 1.5)),  # 改变亮度
        iaa.GaussianBlur(sigma=(0, 0.5)),  # iaa.GaussianBlur(0.5),
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
            rotate=(90, 90)
        )  # 对一部分图像做仿射变换, rotate旋转±30度之间, scale图像缩放为80%到95%之间, translate_px 独立地在x轴和y轴上将图像平移到15像素
    ])

    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下
    # root所指的是当前正在遍历的这个文件夹的本身的地址
    # sub_folders 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    for root, sub_folders, files in os.walk(XML_DIR):
        for name in tqdm(files):
            bndbox = read_xml_annotation(XML_DIR, name)
            if len(bndbox) == 0:
                print("There is no bounding box : ", name)
                continue
            shutil.copy(os.path.join(XML_DIR, name), AUG_XML_DIR)
            shutil.copy(os.path.join(IMG_DIR, name[:-4] + '.jpg'), AUG_IMG_DIR)
            if "anthracnose_runner" in name:
                AUGLOOP = 2
            elif "gray_mold" in name:
                AUGLOOP = 2
            elif "blossom_blight" in name:
                AUGLOOP = 2
            elif "leaf_spot" in name:
                AUGLOOP = 2
            elif "powdery_mildew_fruit" in name:
                AUGLOOP = 2
            elif "powdery_mildew_leaf" in name:
                AUGLOOP = 2
            elif "anthracnose_fruit_rot" in name:
                AUGLOOP = 2
            elif "angular_leafspot" in name:
                AUGLOOP = 2
            else:
                print("wrong?!",name)
                # continue
            for epoch in range(AUGLOOP):
                a = random.randint(0,1)
                if a ==0:
                    seq_det = seq.to_deterministic()
                else:
                    seq_det = seq2.to_deterministic()  # 保持坐标和图像同步改变，而不是随机

                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                # sp = img.size
                img = np.asarray(img)
                # bndbox 坐标增强
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    n_x1 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x1)))
                    n_y1 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y1)))
                    n_x2 = int(max(1, min(img.shape[1], bbs_aug.bounding_boxes[0].x2)))
                    n_y2 = int(max(1, min(img.shape[0], bbs_aug.bounding_boxes[0].y2)))
                    if n_x1 == 1 and n_x1 == n_x2:
                        n_x2 += 1
                    if n_y1 == 1 and n_y2 == n_y1:
                        n_y2 += 1
                    if n_x1 >= n_x2 or n_y1 >= n_y2:
                        print('error', name)
                        SAVE = False
                    new_bndbox_list.append([n_x1, n_y1, n_x2, n_y2])
                if len(new_bndbox_list) == 0:
                    SAVE = False
                if SAVE:
                    # 存储变化后的图片
                    image_aug = seq_det.augment_images([img])[0]
                    path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + str(epoch) + '.jpg')
                    image_auged = bbs.draw_on_image(image_aug, thickness=0)
                    Image.fromarray(image_auged).convert('RGB').save(path)

                    # 存储变化后的XML
                    change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR,
                                            str(name[:-4]) + str(epoch))
                    # print(str(name[:-4]) + str(epoch) + '.jpg')
                    new_bndbox_list = []
                else:
                    print("skip this image, SAVE is ", SAVE)
                    new_bndbox_list = []
                    SAVE = True
