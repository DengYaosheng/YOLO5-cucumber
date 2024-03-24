import numpy as np
import torch
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords


def detect(source,
           model,
           device,
           imgsz,
           augment,
           conf_thres,
           iou_thres,
           classes,
           agnostic_nms):

    half = device.type != 'cpu'
    stride = int(model.stride.max())
    img_sz = check_img_size(imgsz, s=stride)
    if half:
        model.half()
    # img0 = cv2.imread(source)
    img = letterbox(source, img_sz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img, augment=augment)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
    g_list = []
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], source.shape).round()
            for *xyxy, conf, cls in reversed(det):
                line = (cls, *xyxy, conf)
                s = ('%g ' * len(line)).rstrip() % line
                g_list.append(s)
    return g_list
