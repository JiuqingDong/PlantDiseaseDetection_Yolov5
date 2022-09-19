import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

anthracnose_runner =  0
gray_mold =  0
blossom_blight =  0
leaf_spot =  0
powdery_mildew_fruit =  0
powdery_mildew_leaf =  0
anthracnose_fruit_rot =  0
angular_leafspot =  0

color = [255,255,255]

folder = 'dataset_test'
xml_path = '/Users/Dong/Desktop/test2500/' + folder + '/test_xml'     # 你的xml文件路径
img_path = '/Users/Dong/Desktop/test2500/' + folder + '/detect'         # 图像路径
img_xml = '/Users/Dong/Desktop/test2500/' + folder + '/show_test'       # 显示标注框保存该文件的路径
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

            if objectName in ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold",
        "Leaf Spot",  "Powdery Mildew Fruit", "Powdery Mildew Leaf"]:
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (255,255,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),color,box_thickness)
                angular_leafspot+=1

            if objectName not in ["Angular Leafspot", "Anthracnose Fruit Rot", "Blossom Blight", "Gray Mold",
        "Leaf Spot",  "Powdery Mildew Fruit", "Powdery Mildew Leaf"]:
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0, 0, 255), text_thickness)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [64, 128, 256], box_thickness)
                print('objectName not in these labels. It is :', objectName)

            if len(allObjects) == 0:
                print("error")

        name = name.replace('xml', 'jpg')
        img_save_path = os.path.join(img_xml, name)
        print(img_save_path)
        cv2.imwrite(img_save_path, img)
