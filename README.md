This repository includes the official implementation of the paper:

Data-centric Annotation Analysis for Plant Disease Detection: Strategy, Consistency, and Performance.

(State: Under Review)

Authors and affiliations:

Jiuqing Dong 1, Jaehwan Lee1, 2, Alvaro Fuentes 1,2, Sook Yoon 3,, Mun Haeng Lee 4, Dong Sun Park 1,2, 1 Department of Electronic Engineering, Jeonbuk National University, Jeonju, South Korea 2 Core Research Institute of Intelligent Robots, Jeonbuk National University, Jeonju, South Korea 3 Department of Computer Engineering, Mokpo National University, Muan, South Korea 4 Fruit Vegetable Research Institute, Chungnam A.R.E.S, Buyeo, South Korea

The code include YOLO-v5 implement, Noise generation, Data-augmentation by rotation, Visualization module, and Label processing part. In addition, to facilitate further research by practitioners, we will provide the pretrained model. However, due to non-disclosure agreements, we are temporarily unable to make the dataset public.

You also can refer to the Official implement of YOLO-v5.

# Install

Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.7.

cd yolov5

pip install -r requirements.txt  # install

# Train Paprika

python train.py --img 640 --batch 16 --epochs 200 --data data/paprika_v4.yaml --cfg models/yolov5x.yaml --weights weights/yolov5x.pt --device 1 --project paprika_v4/x

--data: the configuration of training.

-- weights: the pre-trained model

--divice: GPU index

--project: a folder for saving output


# Test
python test.py --data data/paprika_v4.yaml --weights paprika_v4/x/exp/weights/best.pt --device 2 --project paprika_v4/x

# Detect
python detect.py --source /home/multiai3/Jiuqing/yolo5-official-new/datasets/paprika_v4/images/test --weights paprika_v4/x/exp/weights/best.pt --device 2 --project paprika_v4/x

--source: source of image. 

# Visualization
python detect_visualization.py --source /home/multiai3/Jiuqing/yolo5-official-new/datasets/paprika_v4/images/test --weights paprika_v4/x/exp/weights/best.pt --device 2 --project paprika_v4/x

--source: source of image. 

# noise generation:
Please refer to ./tools/process_labels/noise*.py

# Data augmentation
Please refer to ./tools/process_images/data_aug_*.py

# Split the datasets
Please refer to ./tools/process_images/split_the_datasets.py

# Modify orientatiion
Please refer to https://medium.com/@ageitgey/the-dumb-reason-your-fancy-computer-vision-app-isnt-working-exif-orientation-73166c7d39da

If you don't have this problem, please ignore this.

Otherwise, please refer to ./tools/process_images/modify_orientation.py



