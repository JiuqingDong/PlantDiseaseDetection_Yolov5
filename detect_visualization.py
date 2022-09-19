import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image

COLORS = np.random.uniform(0, 255, size=(80, 3))

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


def renormalize_cam_in_bounding_boxes_new(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=False)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes

def renormalize_cam_in_bounding_boxes_old(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=False)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


def renormalize_cam_in_bounding_boxes_copy(boxes, colors, names, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1]
    inside every bounding boxes, and zero outside of the bounding boxes. """
#    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
#    for x1, y1, x2, y2 in boxes:
#        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
#    renormalized_cam = scale_cam_image(renormalized_cam)
#    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    #image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    image_with_bounding_boxes = draw_detections(boxes, colors, names, grayscale_cam)

    return image_with_bounding_boxes


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/strawberry.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.5,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='detect',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    pro = str(save_dir)
    cam_save_dir = "/home/multiai3/Jiuqing/yolo5-official-new/" + pro + '_cam/'
    cambox_save_dir = "/home/multiai3/Jiuqing/yolo5-official-new/" + pro + '_cambox/'
    concat_save_dir = "/home/multiai3/Jiuqing/yolo5-official-new/" + pro + '_concat/'

    # if not os.path.exists(cam_save_dir):
    #     os.mkdir(cam_save_dir)
    # if not os.path.exists(cambox_save_dir):
    #     os.mkdir(cambox_save_dir)
    if not os.path.exists(concat_save_dir):
        os.mkdir(concat_save_dir)
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)

    target_layers23 = [model.model.model[-2]]
    cam23 = EigenCAM(model, target_layers23, use_cuda=True)
    #target_layers22 = [model.model.model[-3]]
    #cam22 = EigenCAM(model, target_layers22, use_cuda=True)
    #target_layers21 = [model.model.model[-4]]
    #cam21 = EigenCAM(model, target_layers21, use_cuda=True)
    target_layers20 = [model.model.model[-5]]
    cam20 = EigenCAM(model, target_layers20, use_cuda=True)
    #target_layers19 = [model.model.model[-6]]
    #cam19 = EigenCAM(model, target_layers19, use_cuda=True)
    #target_layers18 = [model.model.model[-7]]
    #cam18 = EigenCAM(model, target_layers18, use_cuda=True)
    target_layers17 = [model.model.model[-8]]
    cam17 = EigenCAM(model, target_layers17, use_cuda=True)
    #target_layers16 = [model.model.model[-9]]
    #cam16 = EigenCAM(model, target_layers16, use_cuda=True)
    #target_layers15 = [model.model.model[-10]]
    #cam15 = EigenCAM(model, target_layers15, use_cuda=True)
    #target_layers14 = [model.model.model[-11]]
    #cam14 = EigenCAM(model, target_layers14, use_cuda=True)
    target_layers13 = [model.model.model[-12]]
    cam13 = EigenCAM(model, target_layers13, use_cuda=True)
    #target_layers12 = [model.model.model[-13]]
    #cam12 = EigenCAM(model, target_layers12, use_cuda=True)
    #target_layers11 = [model.model.model[-14]]
    #cam11 = EigenCAM(model, target_layers11, use_cuda=True)
    #target_layers10 = [model.model.model[-15]]
    #cam10 = EigenCAM(model, target_layers10, use_cuda=True)
    #target_layers9 = [model.model.model[-16]]
    #cam9 = EigenCAM(model, target_layers9, use_cuda=True)
    target_layers8 = [model.model.model[-17]]
    cam8 = EigenCAM(model, target_layers8, use_cuda=True)
    #target_layers7 = [model.model.model[-18]]
    #cam7 = EigenCAM(model, target_layers7, use_cuda=True)
    target_layers6 = [model.model.model[-19]]
    cam6 = EigenCAM(model, target_layers6, use_cuda=True)
    #target_layers5 = [model.model.model[-20]]
    #cam5 = EigenCAM(model, target_layers5, use_cuda=True)
    target_layers4 = [model.model.model[-21]]
    cam4 = EigenCAM(model, target_layers4, use_cuda=True)
    #target_layers3 = [model.model.model[-22]]
    #cam3 = EigenCAM(model, target_layers3, use_cuda=True)
    target_layers2 = [model.model.model[-23]]
    cam2 = EigenCAM(model, target_layers2, use_cuda=True)
    #target_layers1 = [model.model.model[-24]]
    #cam1 = EigenCAM(model, target_layers1, use_cuda=True)

    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        file_name = path.split('/')[-1]
        img = im.transpose((1,2,0))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_0 = img.copy()
        # cv2.imwrite('{}{}'.format(cam_save_dir, file_name), img_0)

        img = np.float32(img) / 255

        transform = transforms.ToTensor()
        tensor = transform(img).unsqueeze(0)

        grayscale_cam23 = cam23(tensor)[0, :, :]
        #grayscale_cam22 = cam22(tensor)[0, :, :]
        #grayscale_cam21 = cam21(tensor)[0, :, :]
        grayscale_cam20 = cam20(tensor)[0, :, :]
        #grayscale_cam19 = cam19(tensor)[0, :, :]
        #grayscale_cam18 = cam18(tensor)[0, :, :]
        grayscale_cam17 = cam17(tensor)[0, :, :]
        #grayscale_cam16 = cam16(tensor)[0, :, :]
        #grayscale_cam15 = cam15(tensor)[0, :, :]
        #grayscale_cam14 = cam14(tensor)[0, :, :]
        grayscale_cam13 = cam13(tensor)[0, :, :]
        #grayscale_cam12 = cam12(tensor)[0, :, :]
        #grayscale_cam11 = cam11(tensor)[0, :, :]
        #grayscale_cam10 = cam10(tensor)[0, :, :]
        #grayscale_cam9  = cam9(tensor)[0, :, :]
        grayscale_cam8  = cam8(tensor)[0, :, :]
        #grayscale_cam7  = cam7(tensor)[0, :, :]
        grayscale_cam6  = cam6(tensor)[0, :, :]
        #grayscale_cam5  = cam5(tensor)[0, :, :]
        grayscale_cam4  = cam4(tensor)[0, :, :]
        #grayscale_cam3  = cam3(tensor)[0, :, :]
        grayscale_cam2  = cam2(tensor)[0, :, :]
        #grayscale_cam1  = cam1(tensor)[0, :, :]


        cam_image23 = show_cam_on_image(img, grayscale_cam23, use_rgb=False)
        #cam_image22 = show_cam_on_image(img, grayscale_cam22, use_rgb=False)
        #cam_image21 = show_cam_on_image(img, grayscale_cam21, use_rgb=False)
        cam_image20 = show_cam_on_image(img, grayscale_cam20, use_rgb=False)
        #cam_image19 = show_cam_on_image(img, grayscale_cam19, use_rgb=False)
        #cam_image18 = show_cam_on_image(img, grayscale_cam18, use_rgb=False)
        cam_image17 = show_cam_on_image(img, grayscale_cam17, use_rgb=False)
        #cam_image16 = show_cam_on_image(img, grayscale_cam16, use_rgb=False)
        #cam_image15 = show_cam_on_image(img, grayscale_cam15, use_rgb=False)
        #cam_image14 = show_cam_on_image(img, grayscale_cam14, use_rgb=False)
        cam_image13 = show_cam_on_image(img, grayscale_cam13, use_rgb=False)
        #cam_image12 = show_cam_on_image(img, grayscale_cam12, use_rgb=False)
        #cam_image11 = show_cam_on_image(img, grayscale_cam11, use_rgb=False)
        #cam_image10 = show_cam_on_image(img, grayscale_cam10, use_rgb=False)
        #cam_image9  = show_cam_on_image(img, grayscale_cam9 , use_rgb=False)
        cam_image8  = show_cam_on_image(img, grayscale_cam8 , use_rgb=False)
        #cam_image7  = show_cam_on_image(img, grayscale_cam7 , use_rgb=False)
        cam_image6  = show_cam_on_image(img, grayscale_cam6 , use_rgb=False)
        #cam_image5  = show_cam_on_image(img, grayscale_cam5 , use_rgb=False)
        cam_image4  = show_cam_on_image(img, grayscale_cam4 , use_rgb=False)
        #cam_image3  = show_cam_on_image(img, grayscale_cam3 , use_rgb=False)
        cam_image2  = show_cam_on_image(img, grayscale_cam2 , use_rgb=False)
        #cam_image1  = show_cam_on_image(img, grayscale_cam1 , use_rgb=False)


        # cam_image_0 = cv2.cvtColor(cam_image_0, cv2.COLOR_BGR2RGB)

        # cv2.imwrite('{}{}'.format(cam_save_dir,file_name), cam_image)

        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0

        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):

                boxes, col, labels = [], [], []
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):
                    xmin = int(xyxy[0])
                    ymin = int(xyxy[1])
                    xmax = int(xyxy[2])
                    ymax = int(xyxy[3])
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    boxes.append((xmin, ymin, xmax, ymax))
                    col.append([128,64,32])
                    labels.append(label)

                try:
                    renormalized_cam_image = renormalize_cam_in_bounding_boxes_new(boxes, col, labels, img, grayscale_cam23)
                except:
                    print("===========", file_name, file_name, file_name)
                else:
                    cv2.imwrite('{}{}'.format(concat_save_dir, file_name), np.hstack((img_0, cam_image2, cam_image4, cam_image6, cam_image8,
                                                                                      cam_image13, cam_image17, cam_image20, cam_image23, renormalized_cam_image)))

                    #line1 = np.hstack((img_0, cam_image1, cam_image2, cam_image3, cam_image4))
                    #line2 = np.hstack((cam_image5, cam_image6, cam_image7, cam_image8, cam_image9))
                    #line3 = np.hstack((cam_image10, cam_image11, cam_image12, cam_image13, cam_image14))
                    #line4 = np.hstack((cam_image15, cam_image16, cam_image17, cam_image18, cam_image19))
                    #line5 = np.hstack((cam_image20, cam_image21, cam_image22, cam_image23, renormalized_cam_image))
                    #cv2.imwrite('{}{}'.format(concat_save_dir, file_name), np.vstack((line1, line2, line3, line4, line5)))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # Write results
                for *xyxy, conf, cls in reversed(det):

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))

                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()  # im0是一张图片
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)

            if save_img:        # save_img = true
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default= '/home/multiai3/Jiuqing/yolo5-official-new/datasets/strawberry_v4_aug/images/test', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', type=str, default= '/home/multiai3/Jiuqing/yolo5-official-new/datasets/paprika_v4/images/test', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/paprika_with_semi_global.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true',help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='detect', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=True, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
