import xml.etree.ElementTree as ET
import cv2
import os
from tqdm import tqdm

blossom_end_rot =  0
graymold =  0
powdery_mildew =  0
spider_mite =  0
spotting_disease =  0


folder = 'dataset_test'
xml_path = '/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels/train_xml'     # 你的xml文件路径
img_path = '/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/images/train'         # 图像路径

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

            if objectName == 'blossom_end_rot':     # 把引号里的内容更改为自己的类别即可。
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255,255,255), text_thickness)
                #cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                blossom_end_rot+=1

            if objectName == 'graymold':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255,255,255), text_thickness)
                #cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                graymold+=1

            if objectName == 'powdery_mildew':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255,255,255), text_thickness)
                #cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                powdery_mildew+=1

            if objectName == 'spider_mite':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255,255,255), text_thickness)
                #cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                spider_mite+=1

            if objectName == 'spotting_disease':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255,255,255), text_thickness)
                #cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                spotting_disease+=1

            if objectName not in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                #cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                #            (255, 255, 255), text_thickness)
                #cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [0, 0, 0], box_thickness)
                print('objectName not in these labels. It is :', objectName)
            if len(allObjects) == 0:
                print("error")

        #name = name.replace('xml', 'jpg')
        #img_save_path = os.path.join(img_xml, name)
        #cv2.imwrite(img_save_path, img)

print("blossom_end_rot boxs is :", blossom_end_rot)
print("graymold is :", graymold)
print("powdery_mildew is :", powdery_mildew)
print("spider_mite is :", spider_mite)
print("spotting_disease boxs is :", spotting_disease)
