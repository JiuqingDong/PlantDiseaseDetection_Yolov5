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

folder = 'dataset_org'
xml_path = '/Users/Dong/Desktop/test2500/' + folder + '/train_xml'     # 你的xml文件路径
img_path = '/Users/Dong/Desktop/test2500/' + folder + '/train'         # 图像路径
img_xml = '/Users/Dong/Desktop/test2500/' + folder + '/show_xml'       # 显示标注框保存该文件的路径
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
            '''
            if objectName == 'blossom_end_rot':     # 把引号里的内容更改为自己的类别即可。
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (255,255,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,0],box_thickness)

            if objectName == 'graymold':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (255,0,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,0,255],box_thickness)

            if objectName == 'powdery_mildew':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (0,255,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[0,255,255],box_thickness)

            if objectName == 'spider_mite':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (255,0,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,0,0],box_thickness)

            if objectName == 'spotting_disease':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (0,0,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[0,0,255],box_thickness)

            if objectName not in ['blossom_end_rot', 'graymold', 'powdery_mildew', 'spider_mite', 'spotting_disease']:
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size,
                            (0, 0, 255), text_thickness)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), [0, 0, 0], box_thickness)
                print('objectName not in these labels. It is :', objectName)

            '''

            if objectName == 'Angular Leafspot':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (255,255,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,0],box_thickness)
                angular_leafspot+=1

            if objectName == 'Anthracnose Fruit Rot':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (255,0,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,0,255],box_thickness)
                anthracnose_fruit_rot+=1

            if objectName == 'Blossom Blight':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (255,0,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,0,0],box_thickness)
                blossom_blight+=1

            if objectName == 'Gray Mold':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,0,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[0,0,255],box_thickness)
                gray_mold+=1

            if objectName == 'Leaf Spot':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (255,255,255), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[0,255,0],box_thickness)
                leaf_spot+=1

            if objectName == 'Powdery Mildew Fruit':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,255,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[0,0,0],box_thickness)
                powdery_mildew_fruit+=1

            if objectName == 'Powdery Mildew Leaf':
                xmin = int(object.find('bndbox').find('xmin').text)
                ymin = int(object.find('bndbox').find('ymin').text)
                xmax = int(object.find('bndbox').find('xmax').text)
                ymax = int(object.find('bndbox').find('ymax').text)
                cv2.putText(img, objectName, (xmin, ymax), font, text_size, (0,0,0), text_thickness)
                cv2.rectangle(img,(xmin, ymin),(xmax, ymax),[255,255,255],box_thickness)
                powdery_mildew_leaf+=1

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
        #cv2.imshow(name, img)
        #cv2.waitKey(1000)
        #cv2.destroyAllWindows()
        name = name.replace('xml', 'jpg')
        img_save_path = os.path.join(img_xml, name)
        # print(img_save_path)
        cv2.imwrite(img_save_path, img)

print("anthracnose_runner boxs is :", anthracnose_runner)
print("gray_mold boxs is :", gray_mold)
print("blossom_blight boxs is :", blossom_blight)
print("leaf_spot boxs is :", leaf_spot)
print("powdery_mildew_fruit boxs is :", powdery_mildew_fruit)
print("powdery_mildew_leaf boxs is :", powdery_mildew_leaf)
print("anthracnose_fruit_rot boxs is :", anthracnose_fruit_rot)
print("angular_leafspot boxs is :", angular_leafspot)
'''
anthracnose_runner
gray_mold
blossom_blight
leaf_spot
powdery_mildew_fruit
powdery_mildew_leaf
anthracnose_fruit_rot
angular_leafspot
'''