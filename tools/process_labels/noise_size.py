import os
import random


def concat_ratio(rate):
    x_rate = random.randint(-rate, rate)/100
    y_rate = random.randint(-rate, rate)/100
    w_rate = random.randint(-rate, rate)/100
    h_rate = random.randint(-rate, rate)/100
    return x_rate, y_rate, w_rate, h_rate


def norm(noise):
    if noise<0:
        noise = 0.00000001
        print("norm")
    if noise>1:
        noise = 0.99999999
        print("norm")

    return noise


def add_size_noise(info, Rate):
    x = float(info[1])
    y = float(info[2])
    w = float(info[3])
    h = float(info[4])
    x_rate, y_rate, w_rate, h_rate = concat_ratio(rate=Rate)

    w_noise = w + w * w_rate
    h_noise = h + h * h_rate

    if (x < w_noise/2) or ((1-x) < w_noise/2):
        w_noise = w
        print("x")
    if (y < h_noise/2) or ((1-y) < h_noise/2):
        h_noise = h
        print("y")

    w_noise = str(norm(w_noise))
    h_noise = str(norm(h_noise))

    info[3] = w_noise
    info[4] = h_noise

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
                    position_noise = add_size_noise(row, rate)
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

#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise5',Rate=5)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise10',Rate=10)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise15',Rate=15)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise20',Rate=20)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise25',Rate=25)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise30',Rate=30)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise35',Rate=35)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise40',Rate=40)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise45',Rate=45)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4_aug/labels_ori_txt/', target='labels_size_noise50',Rate=50)



#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise5',Rate=5)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise10',Rate=10)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise15',Rate=15)
#run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise20',Rate=20)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise25',Rate=25)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise30',Rate=30)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise35',Rate=35)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise40',Rate=40)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise45',Rate=45)
run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_ori_txt/', target='labels_size_noise50',Rate=50)
