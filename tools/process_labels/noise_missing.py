import os
import random


def missing_ratio(rate):
    a = random.randint(0, 99)
    noise = False
    if a < rate:
        noise = True
    return noise


def noise_txt(txt_file, target, rate):
    target_file = txt_file.replace('labels_ori_txt', target)

    with open(target_file, 'w+') as saver:
        with open(txt_file, 'r+') as file:
            lines = file.readlines()
            file.seek(0, 0) #set the pointer to 0,0 cordinate of file
            for line in lines:
                missing_label = missing_ratio(rate)
                if missing_label is not True:
                    saver.write(line)
    saver.close()


def run(txt_path='/Volumes/Data/nosie_labels/', target = None,  Rate=5):
    for roots, dirs, files in os.walk(txt_path):
        target_dir = roots.replace('labels_ori_txt', target)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(roots, file)
                noise_txt(txt_file=file_path, target = target, rate=Rate)
    print("Noise rate {} Done!".format(Rate))


run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise5',Rate=5)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise10',Rate=10)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise15',Rate=15)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise20',Rate=20)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise25',Rate=25)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise30',Rate=30)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise35',Rate=35)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise40',Rate=40)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise45',Rate=45)
run(txt_path='/Users/Dong/Desktop/paprika_v4/labels_ori_txt', target='labels_missing_noise50',Rate=50)