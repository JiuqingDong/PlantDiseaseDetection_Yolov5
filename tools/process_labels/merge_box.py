import xml.etree.ElementTree as ET
from lxml import etree
import os
import numpy


def get_value(category, xmin, xmax, ymin, ymax):
    category = str(category.text)
    xmin = int(xmin.text)
    xmax = int(xmax.text)
    ymin = int(ymin.text)
    ymax = int(ymax.text)

    return category, xmin, xmax, ymin, ymax


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


def min_max(list):
    list = numpy.array(list)
    if len(list) == 1:
        xmin = list[0,0]
        xmax = list[0,1]
        ymin = list[0,2]
        ymax = list[0,3]
    else:
        xmin = min(list[:,0])
        xmax = max(list[:,1])
        ymin = min(list[:,2])
        ymax = max(list[:,3])
    return str(xmin),str(xmax),str(ymin),str(ymax)

def add_node(root,label,xmin,ymin,xmax,ymax):
    object = etree.Element("object")
    namen =  etree.SubElement(object,"name")
    namen.text = label
    object.append(namen)
    pose = etree.SubElement(object,"pose")
    pose.text = str(0)
    object.append(pose)
    truncated = etree.SubElement(object,"truncated")
    truncated.text = str(0)
    object.append(truncated)
    difficult = etree.SubElement(object,"difficult")
    difficult.text = str(0)
    object.append(difficult)
    bndbox = etree.SubElement(object,"bndbox")
    xminn = etree.SubElement(bndbox,"xmin")
    xminn.text = str(xmin)
    bndbox.append(xminn)
    yminn = etree.SubElement(bndbox,"ymin")
    yminn.text = str(ymin)
    bndbox.append(yminn)
    xmaxn = etree.SubElement(bndbox,"xmax")
    xmaxn.text = str(xmax)
    bndbox.append(xmaxn)
    ymaxn = etree.SubElement(bndbox,"ymax")
    ymaxn.text = str(ymax)
    root.getroot().append(object)
    
    
def merge_xml(xml_file, target):
    target_file = xml_file.replace('labels', target)

    #tree = ET.parse(xml_file)
    #root = tree.getroot()

    parser = etree.XMLParser(remove_blank_text=True)  #
    root = etree.parse(xml_file, parser)
    tree = etree.ElementTree(root.getroot())

    node = etree.Element('root')


    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    all_objects = {"blossom_end_rot":[], "graymold":[], "spotting_disease":[]}

    for object in tree.findall('object'):
        category = object.find('name')
        bndbox = object.find('bndbox')  # 子节点下节点rank的值
        xmin = bndbox.find('xmin')       # type(xmin) = int
        xmax = bndbox.find('xmax')
        ymin = bndbox.find('ymin')
        ymax = bndbox.find('ymax')

        old_category, old_xmin, old_xmax, old_ymin, old_ymax = get_value(category, xmin, xmax, ymin, ymax)

        if old_category in ["blossom_end_rot", "graymold", "spotting_disease"]:
            all_objects[old_category].append([old_xmin, old_xmax, old_ymin, old_ymax])
            parent = object.getparent()
            parent.remove(object)

    for dict in ["blossom_end_rot", "graymold", "spotting_disease"]:
        #
        if len(all_objects[dict]) == 0:
            continue
        else:
            xmin, xmax, ymin, ymax = min_max(all_objects[dict])
            add_node(root, dict, xmin, ymin, xmax, ymax)


    tree.write(target_file, pretty_print=True, xml_declaration=False, encoding='utf-8')


def run(xml_path= None, target = None):
    for roots, dirs, files in os.walk(xml_path):
        print(roots)
        target_dir = roots.replace('labels', target)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for file in files:
            if file.endswith('.xml'):
                file_path = os.path.join(roots, file)
                merge_xml(xml_file=file_path, target = target)

    print("Done!")


run(xml_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_xmlnoise/labels/', target='labels_with_semi-global')

