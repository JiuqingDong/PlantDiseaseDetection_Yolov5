import os
import random


def noise_ratio(rate):
    a = random.randint(0, 99)
    noise = False
    if a < rate:
        noise = True
    return noise


def noise_label(org_label):
    if org_label == '0':
        new_label = random.choice('1234')
    elif org_label == '1':
        new_label = random.choice('0234')
    elif org_label == '2':
        new_label = random.choice('0134')
    elif org_label == '3':
        new_label = random.choice('0124')
    else:
        new_label = random.choice('0123')
    return new_label


def noise_txt(txt_file, rate):
    with open(txt_file, 'r+') as file:
        lines = file.readlines()
        file.seek(0, 0) #set the pointer to 0,0 cordinate of file
        for line in lines:
            row = line.strip().split(" ")
            noise = noise_ratio(rate)
            if noise:
                wrong_label = noise_label(row[0])
                print("The label is changed from {} to {}.".format(row[0], wrong_label))
                row[0] = wrong_label
            file.write(" ".join(row) + "\n")


def run(txt_path='/Volumes/Data/nosie_labels/', Rate=5):

    for roots, dirs, files in os.walk(txt_path):
        print(roots)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(roots, file)
                noise_txt(txt_file=file_path, rate=Rate)

    print("Noise rate {} Done!".format(Rate))

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise5/', Rate=5)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise10/', Rate=15)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise15/', Rate=15)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise20/.', Rate=20)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise25/.', Rate=25)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise30/.', Rate=30)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise35/.', Rate=35)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise40/.', Rate=40)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise45/.', Rate=45)

run(txt_path='/Volumes/Data/project_paper_code_data/yolov5-official_new/datasets/paprika_v4/labels_noise50/.', Rate=50)


