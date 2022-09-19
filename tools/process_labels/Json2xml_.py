import codecs
import json
import os


def transfer_category(classes):
    category = None
    if classes == 'a9':
        category = 'powdery_mildew'
    elif classes == 'a10':
        category = 'powdery_mildew'
    elif classes == 'b3':
        category = 'phy_disorder'
    elif classes == 'b6':
        category = 'phy_disorder'
    elif classes == 'b7':
        category = 'phy_disorder'
    elif classes == 'b8':
        category = 'phy_disorder'
    elif classes == '00':
        category = 'normal_fruit'
    elif classes == 'c9':
        category = 'chem_damage'
    else:
        print(classes)
        print("wrong!!!!!!!!!!!!!!!!!!")
    return category


def read_json(json_file):
    with open(json_file, 'r') as jf:
        jf_data = json.load(jf)
        name = jf_data['description']['image']
        height = jf_data['description']['height']
        width = jf_data['description']['width']
        if len(jf_data['annotations']['bbox']) != 1:
            print(json_file)
            print(len(jf_data['annotations']['bbox']))
        xmin = int(jf_data['annotations']['bbox'][0]['x'])
        ymin = int(jf_data['annotations']['bbox'][0]['y'])
        xmax = int(jf_data['annotations']['bbox'][0]['w'] + xmin)
        ymax = int(jf_data['annotations']['bbox'][0]['h'] +ymin)
        category = str(jf_data['annotations']['disease'])
        category = transfer_category(classes = category)
        # print(name, xmin, ymin, xmax, ymax, category,height,width)
    return name, xmin, ymin, xmax, ymax, category,height,width


def xml_write(xml_file, json_file):
    jpg_file = json_file.replace('.json', '.jpg')
    jpg_file = jpg_file.replace('/label', '/image')
    name, xmin, ymin, xmax, ymax, category, height, width = read_json(json_file)
    if category not in ['powdery_mildew',
                        'phy_disorder',
                        'chem_damage',
                        'normal_fruit']:
        print(category)
        print(json_file)
    with codecs.open(xml_file, "w", "utf-8") as xml:
        xml.write('<annotation>\n')
        xml.write('\t<folder>'+'test'+'</folder>\n')
        xml.write('\t<filename>'+name+'</filename>\n')
        xml.write('\t<path>'+jpg_file+'</path>\n')
        xml.write('\t<source>\n')
        xml.write('\t\t<database>Unknown</database>\n')
        xml.write('\t</source>\n')
        xml.write('\t<size>\n')
        xml.write('\t\t<width>'+str(width)+'</width>\n')
        xml.write('\t\t<height>'+str(height)+'</height>\n')
        xml.write('\t\t<depth>3</depth>\n')
        xml.write('\t</size>\n')
        xml.write('\t<segmented>0</segmented>\n')
        if xmax<=xmin:
            pass
        elif ymax<=ymin:
            pass
        else:
            xml.write('\t<object>\n')
            xml.write('\t\t<name>'+category+'</name>\n')
            xml.write('\t\t<pose>Unspecified</pose>\n')
            xml.write('\t\t<truncated>0</truncated>\n')
            xml.write('\t\t<difficult>0</difficult>\n')
            xml.write('\t\t<bndbox>\n')
            xml.write('\t\t\t<xmin>' + str(xmin) + '</xmin>\n')
            xml.write('\t\t\t<ymin>' + str(ymin) + '</ymin>\n')
            xml.write('\t\t\t<xmax>' + str(xmax) + '</xmax>\n')
            xml.write('\t\t\t<ymax>' + str(ymax) + '</ymax>\n')
            xml.write('\t\t</bndbox>\n')
            xml.write('\t</object>\n')
        xml.write('</annotation>')

def main(json_fold = '/Users/jiuqingdong/Desktop/dataset_examples/strawberry/labels/'):
    for roots, dirs, files in os.walk(json_fold):
        for file in files:
            if file.endswith('.json'):
                json_file = os.path.join(roots, file)
                xml_roots = roots.replace('labels', 'xmls')
                print(xml_roots)
                if not os.path.exists(xml_roots):
                    os.mkdir(xml_roots)
                name = file.split('.')[0]+'.xml'
                xml_file = os.path.join(xml_roots, name)
                xml_write(xml_file, json_file)

main()