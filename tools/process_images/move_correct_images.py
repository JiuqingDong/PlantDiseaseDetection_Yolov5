import os
import shutil
#project = 'strawberry_bigbox_v5'
project = 'paprika_semi_global/x'

detect_path = '/home/multiai3/Jiuqing/yolo5-official-new/'+project+'/detect'
txt_path = '/home/multiai3/Jiuqing/yolo5-official-new/'+project+'/detect_wrong.txt'

with open(txt_path, 'r') as wrong_list:
    wrong = wrong_list.read().splitlines()
wrong_path = detect_path+'_wrong'

if not os.path.exists(wrong_path):
    os.mkdir(wrong_path)
i = 0
for root, dirs, files in os.walk(detect_path):
    for file in files:
        if file in wrong:
            src = os.path.join(root,file)
            shutil.move(src, wrong_path)
            i+=1
print(i)
