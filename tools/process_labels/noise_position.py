import os
import random


def position_ratio(rate):
    x_rate = random.randint(-rate, rate)/100
    y_rate = random.randint(-rate, rate)/100
    return x_rate, y_rate


def norm(noise):
    if noise<0:
        noise = 0.00000001
        print("norm")
    if noise>1:
        noise = 0.99999999
        print("norm")

    return noise


def add_position_noise(info, Rate):
    x = float(info[1])
    y = float(info[2])
    w = float(info[3])
    h = float(info[4])
    x_rate, y_rate = position_ratio(rate=Rate)

    x_noise = x + w * x_rate
    y_noise = y + h * y_rate

    if (x_noise < w/2) or ((1-x_noise) < w/2):
        x_noise = x
        print("x")
    if (y_noise < h/2) or ((1-y_noise) < h/2):
        y_noise = y
        print("y")

    x_noise = str(norm(x_noise))
    y_noise = str(norm(y_noise))

    info[1] = x_noise
    info[2] = y_noise

    return info


def noise_txt(txt_file, target, rate):
    target_file = txt_file.replace('labels_ori_txt', target)
    with open(target_file, 'w+') as saver:
        with open(txt_file, 'r+') as file:
            lines = file.readlines()
            file.seek(0, 0) #set the pointer to 0,0 cordinate of file
            for line in lines:
                row = line.strip().split(" ")
                if len(row) == 5:
                    position_noise = add_position_noise(row, rate)
                    saver.write(" ".join(position_noise) + "\n")
                else:
                    print(row)
    saver.close()

def run(txt_path='/Volumes/Data/nosie_labels/', target = None,  Rate=5):
    for roots, dirs, files in os.walk(txt_path):
        print(roots)
        target_dir = roots.replace('labels_ori_txt', target)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(roots, file)
                noise_txt(txt_file=file_path, target = target, rate=Rate)

    print("Noise rate {} Done!".format(Rate))

#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori/', target='labels_position_noise5',Rate=5)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori/', target='labels_position_noise10',Rate=10)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori/', target='labels_position_noise15',Rate=15)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori/', target='labels_position_noise20',Rate=20)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise25',Rate=25)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise30',Rate=30)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise35',Rate=35)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise40',Rate=40)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise45',Rate=45)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_position_noise50',Rate=50)


#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise5',Rate=5)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise10',Rate=10)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise15',Rate=15)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise20',Rate=20)


run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise25',Rate=25)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise30',Rate=30)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise35',Rate=35)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise40',Rate=40)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise45',Rate=45)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_position_noise50',Rate=50)