#!/usr/bin/python3
import yaml
import rospy
import cv2
import sys
import numpy as np
import torch
from pathlib import Path
from cv_bridge import CvBridge
from typing import Tuple, Union, List
from sensor_msgs.msg import Image, CompressedImage
from darknet_ros_msgs.msg import BoundingBox, BoundingBoxes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1] / "ultralytics"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import ops
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.augment import LetterBox
from ultralytics.utils.torch_utils import select_device

class YOLOv8Detector:
    def __init__(self, model: str, device: str, data: str, view_image: bool, hide_label: bool,
                 img_topic: str, pub_topic1: str, pub_topic2: str, single_cls: Union[str, None],
                 imgsz: Union[Tuple[int, int], None] = (480, 640), conf_thresh: float = 0.5,
                 iou_thresh: float = 0.45, done_warmup: bool = False,
                 ):

        with open(data, 'r') as f:
            yml_content = yaml.load(f, Loader=yaml.FullLoader)
        self.cls_dict = yml_content['names']
        self.color_palette = np.random.uniform(
            0, 255, size=(len(self.cls_dict), 3))
        if single_cls != "":
            single_cls_id = [
                key for key, value in self.cls_dict.items() if value == single_cls][0]
            self.single_cls = [int(single_cls_id)]
        else:
            self.single_cls = None
        self.model = model
        self.device = device
        self.iou = iou_thresh
        self.conf = conf_thresh
        self.view_img = view_image
        self.hide_lbl = hide_label
        self.bridge = CvBridge()
        self.model = AutoBackend(model, device=select_device(
            self.device), dnn=True, data=data, fp16=False, fuse=True)
        self.device = self.model.device  # update device
        self.half = self.model.fp16  # update half
        self.model.eval()
        self.imgsz = check_imgsz(imgsz, stride=self.model.stride, min_dim=2)
        if not done_warmup:
            self.model.warmup(imgsz=(
                1 if self.model.pt or self.model.triton else self.dataset.bs, 3, *self.imgsz))
            done_warmup = True
        self.img_sub = rospy.Subscriber(img_topic, Image, self.callback)
        self.det_pub = rospy.Publisher(pub_topic1, BoundingBoxes, queue_size=1)
        self.img_pub = rospy.Publisher(pub_topic2, Image, queue_size=1)

        print("YOLOv8 is ready")

    # def tran2rawsize(self, det, imw, imh):
    #     Kw = imw / 640
    #     Kh = imh / 480
    #     print(imw, imh)
    #     det[:, 0] = det[:, 0] * Kw
    #     det[:, 1] = det[:, 1] * Kh
    #     det[:, 2] = det[:, 2] * Kw
    #     det[:, 3] = det[:, 3] * Kh
    #     return det
    
    def tran2rawsize(self, det_box, im_w, im_h):
        K_w = im_w / 640
        K_h = im_h / 480
        result_box = [
            round(K_w * det_box[0]),
            round(K_h * det_box[1]),
            round(K_w * det_box[2]),
            round(K_h * det_box[3])
        ]
        return result_box

    def preprocess(self, im):
        im0 = im.copy()
        im = cv2.resize(im, (640, 480))
        im = np.stack([LetterBox(self.imgsz, auto=self.model.pt,
                      stride=self.model.stride)(image=im)])
        # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im)
        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0~255 to 0.0~1.0
        return img, im0

    def callback(self, stream):
        self.img_pub.publish(stream)
        im = self.bridge.imgmsg_to_cv2(stream, "bgr8")
        # im = self.bridge.compressed_imgmsg_to_cv2(stream, "bgr8")
        img, im0 = self.preprocess(im)
        img = img[None] if len(img.shape) == 3 else img
        preds = self.model(img, augment=False, visualize=False)
        preds = ops.non_max_suppression(
            preds, self.conf, self.iou, classes=self.single_cls,
            # number of classes
            agnostic=False, max_det=100, nc=len(self.model.names),
        )
        det = preds[0].cpu().numpy()
        bbs = BoundingBoxes()
        bbs.header = stream.header
        bbs.image_header = stream.header
        if len(det):
            # Rescale boxes from imgsz to im0's size
            # det[:, :4] = self.tran2rawsize(det[:, :4], int(im.shape[1]), int(im.shape[0]))
            for *xyxy, conf, cls in reversed(det):
                bb = BoundingBox()
                cls_id = int(cls)
                bb.Class = self.cls_dict[cls_id]  # clsid to cls name
                bb.probability = conf
                bb.xmin = int(xyxy[0])
                bb.ymin = int(xyxy[1])
                bb.xmax = int(xyxy[2])
                bb.ymax = int(xyxy[3])
                bbs.bounding_boxes.append(bb)
                if self.view_img:
                    result_bbx = self.tran2rawsize(xyxy, int(im.shape[1]), int(im.shape[0]))
                    color = self.color_palette[cls_id]
                    # draw box label
                    if self.hide_lbl:
                        label = False
                    else:
                        label = f"{self.cls_dict[cls_id]} {int(conf*100)}%"
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        label_x = result_bbx[0]
                        label_y = result_bbx[1] - 10 if result_bbx[1] - 10 > label_height else result_bbx[1] + 10
                        cv2.rectangle(im0, (label_x, label_y-label_height), (label_x+label_width, label_y+label_height), color, cv2.FILLED)
                        cv2.putText(im0, label, (label_x+3, label_y+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    # draw bounding box
                    cv2.rectangle(im0, (result_bbx[0], result_bbx[1]), (result_bbx[2], result_bbx[3]), color, 2)

        self.det_pub.publish(bbs)  # bbs are based on original images' size

        if self.view_img:
            cv2.imshow('ALeafBoatOnTheHorizon', im0)
            cv2.waitKey(1)


if __name__ == '__main__':

    rospy.init_node('yolov8_ros')

    detector = YOLOv8Detector(
        model=rospy.get_param("~model"),
        device=rospy.get_param("~device"),
        data=rospy.get_param("~class_yaml"),
        view_image=rospy.get_param("~view_image"),
        hide_label=rospy.get_param("~hide_label"),
        img_topic=rospy.get_param("~sub"),
        pub_topic1=rospy.get_param("~pub1"),
        pub_topic2=rospy.get_param("~pub2"),
        # A list of class indices to consider. If None, all classes will be considered.
        single_cls=rospy.get_param("~single_class"),
        conf_thresh=rospy.get_param("~conf_thresh"),
        iou_thresh=rospy.get_param("~iou_thresh"),
    )

    rospy.spin()
