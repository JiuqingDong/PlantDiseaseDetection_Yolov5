import xml.etree.ElementTree as ET
import random
import os

def noise_ratio(rate):
    a = random.randint(0, 99)
    noise = False
    if a < rate:
        noise = True
    return noise


def get_value(category, xmin, xmax, ymin, ymax):
    category = str(category.text)
    xmin = int(xmin.text)
    xmax = int(xmax.text)
    ymin = int(ymin.text)
    ymax = int(ymax.text)

    return category, xmin, xmax, ymin, ymax

"blossom_end_rot", "graymold", "powdery_mildew", "spider_mite","spotting_disease"
def add_category_noise(org_label):
    if org_label == "blossom_end_rot":
        new_label = random.choice(["graymold", "powdery_mildew", "spider_mite", "spotting_disease"])
    elif org_label == "graymold":
        new_label = random.choice(["blossom_end_rot", "powdery_mildew", "spider_mite", "spotting_disease"])
    elif org_label == "powdery_mildew":
        new_label = random.choice(["blossom_end_rot", "graymold",  "spider_mite", "spotting_disease"])
    elif org_label == "spider_mite":
        new_label = random.choice(["blossom_end_rot", "graymold", "powdery_mildew","spotting_disease"])
    elif org_label == "spotting_disease":
        new_label = random.choice(["blossom_end_rot", "graymold", "powdery_mildew", "spider_mite"])
    else:
        print("Wrong")
    return new_label


def concat_ratio(rate):
    x_rate = random.randint(-rate, rate)/100
    y_rate = random.randint(-rate, rate)/100
    w_rate = random.randint(-rate, rate)/100
    h_rate = random.randint(-rate, rate)/100
    return x_rate, y_rate, w_rate, h_rate


def add_concat_noise(old_xmin, old_xmax, old_ymin, old_ymax, rate):
    x = (old_xmin + old_xmax) / 2
    y = (old_ymin + old_ymax) / 2
    w = old_xmax - old_xmin
    h = old_ymax - old_ymin
    x_rate, y_rate, w_rate, h_rate = concat_ratio(rate=rate)
    x_noise = x + w * x_rate
    y_noise = y + h * y_rate
    w_noise = w + w * w_rate
    h_noise = h + h * h_rate

    new_xmin = int(x_noise - (w_noise/2))
    new_xmax = int(x_noise + (w_noise/2))
    new_ymin = int(y_noise - (h_noise/2))
    new_ymax = int(y_noise + (h_noise/2))

    return new_xmin, new_xmax, new_ymin, new_ymax


def norm(new_xmin, new_xmax, new_ymin, new_ymax, width, height):

    if new_xmin < 0:
        new_xmin = 1
    if new_xmax >= width:
        new_xmax = width - 1
    if new_ymin < 0:
        new_ymin = 1
    if new_ymax >= height:
        new_ymax = height - 1

    return new_xmin, new_xmax, new_ymin, new_ymax


def noise_xml(xml_file, target, rate):
    target_file = xml_file.replace('labels', target)

    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)


    for object in root.findall('object'):
        category = object.find('name')
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        xmin = bndbox.find('xmin')       # type(xmin) = int
        xmax = bndbox.find('xmax')
        ymin = bndbox.find('ymin')
        ymax = bndbox.find('ymax')

        old_category, old_xmin, old_xmax, old_ymin, old_ymax = get_value(category, xmin, xmax, ymin, ymax)
        new_xmin, new_xmax, new_ymin, new_ymax = add_concat_noise(old_xmin, old_xmax, old_ymin, old_ymax, rate=rate)
        new_xmin, new_xmax, new_ymin, new_ymax = norm(new_xmin, new_xmax, new_ymin, new_ymax, width, height)
        noise_class = noise_ratio(rate=rate)
        if noise_class:
            new_cotegory = add_category_noise(old_category)
            category.text = str(new_cotegory)
        xmin.text = str(new_xmin)
        xmax.text = str(new_xmax)
        ymin.text = str(new_ymin)
        ymax.text = str(new_ymax)

    tree.write(target_file)

def run(xml_path= None, target = None, Rate = None):
    for roots, dirs, files in os.walk(xml_path):
        print(roots)
        target_dir = roots.replace('labels', target)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(roots, file)
                noise_xml(xml_file=file_path, target = target, rate=Rate)

    print("Noise rate {} Done!".format(Rate))

#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori/', target='labels_concat_noise5',Rate=5)
# run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise5', Rate=5)
# run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise10', Rate=10)
# run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise15', Rate=15)
# run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise20', Rate=20)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise25', Rate=25)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise30', Rate=30)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise35', Rate=35)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise40', Rate=40)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise45', Rate=45)
run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_concat_noise50', Rate=50)
