import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

xml_path = '/Volumes/Data/project_paper_code_data/Plant_diseases_dataset/tomoto_v3/labels/'     # 你的xml文件路径
img_path = '/Volumes/Data/project_paper_code_data/Plant_diseases_dataset/tomoto_v3/JPEGImages/'         # 图像路径
img_xml   = '/Volumes/Data/project_paper_code_data/Plant_diseases_dataset/tomoto_v3/show_xml/'       # 显示标注框保存该文件的路径

all_class = {"healthy": 0, "canker": 1, "plague": 2, "back": 3, "yculr": 4, "miner": 5, "lmold": 6, "tocv": 7, "stress": 8, "powder": 9,
            "gmold": 10, "wfly": 11, "eggfly": 12, "death": 13, "spot": 14,"old": 15, "magdef": 16, "unknown": 17, "wilt": 18}
all_instance = []
# for set in ['train', 'val', 'test']:
#     xml_path = xml_path0 + set +'_xml'
#     img_path = img_path0 + set
for name in tqdm(os.listdir(xml_path)):
    image_name = os.path.join(img_path, name.split('.')[0] + '.jpg')
    if os.path.exists(image_name):
        # 打开xml文档
        tree = ET.parse(os.path.join(xml_path,name))
        img = cv2.imread(image_name)
        box_thickness = int((img.shape[0] + img.shape[1])/1000) + 1  # 标注框的一个参数。本人图像大小不一致，在不同大小的图像上展示不同粗细的bbox
        text_thickness = box_thickness
        text_size = float(text_thickness/2)   # 显示标注类别的参数。字体大小。这些不是重点。不想要可以删掉。
        font = cv2.FONT_HERSHEY_SIMPLEX
        # 得到文档元素对象
        root = tree.getroot()
        allObjects = root.findall('object')
        if len(allObjects) == 0:
            print("1 :", name)
            _image_name = str(image_name)
            _xml_name = _image_name.replace('.jpg', '.xml')
            _xml_name = _xml_name.replace('train', 'train_xml')
            if os.path.exists(_image_name):
                os.remove(_image_name)
            else:
                print("=========", _image_name)
            if os.path.exists(_xml_name):
                os.remove(_xml_name)
            else:
                print("=========", _xml_name)
            continue
        for i in range(len(allObjects)):    # 遍历xml标签，画框并显示类别。
            object = allObjects[i]
            objectName = object.find('name').text
            all_instance.append(objectName)
            xmin = int(object.find('bndbox').find('xmin').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymax = int(object.find('bndbox').find('ymax').text)
            cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,0,0), text_thickness)
            cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
            if len(allObjects) == 0:
                print("error")
        name = name.replace('xml', 'jpg')
        img_save_path = os.path.join(img_xml, name)
        cv2.imwrite(img_save_path, img)

for key in all_class:
    print(key, all_instance.count(key))

'''
for name in tqdm(os.listdir(xml_path0)):
    image_name = os.path.join(img_path0, name.split('.')[0] + '.jpg')

    if os.path.exists(image_name):
        # 打开xml文档
        tree = ET.parse(os.path.join(xml_path0, name))
        img = cv2.imread(image_name)
        box_thickness = int((img.shape[0] + img.shape[1])/1000) + 1  # 标注框的一个参数。本人图像大小不一致，在不同大小的图像上展示不同粗细的bbox

        text_thickness = box_thickness
        text_size = float(text_thickness/2)   # 显示标注类别的参数。字体大小。这些不是重点。不想要可以删掉。
        font = cv2.FONT_HERSHEY_SIMPLEX

        # 得到文档元素对象
        root = tree.getroot()
        allObjects = root.findall('object')
        if len(allObjects) == 0:
            print("1 :", name)
            _image_name = str(image_name)
            _xml_name = _image_name.replace('.jpg', '.xml')
            _xml_name = _xml_name.replace('train', 'train_xml')
            if os.path.exists(_image_name):
                os.remove(_image_name)
            else:
                print("=========", _image_name)
            if os.path.exists(_xml_name):
                os.remove(_xml_name)
            else:
                print("=========", _xml_name)
            continue

        for i in range(len(allObjects)):    # 遍历xml标签，画框并显示类别。
            object = allObjects[i]
            objectName = object.find('name').text

            xmin = int(object.find('bndbox').find('xmin').text)
            ymin = int(object.find('bndbox').find('ymin').text)
            xmax = int(object.find('bndbox').find('xmax').text)
            ymax = int(object.find('bndbox').find('ymax').text)
            cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,0,0), text_thickness)
            cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)

            if len(allObjects) == 0:
                print("error")

        name = name.replace('xml', 'jpg')
        img_save_path = os.path.join(img_xml, name)
        cv2.imwrite(img_save_path, img)
'''