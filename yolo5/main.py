import os
import cv2
import torch
import random
import numpy as np

import copy
import argparse

from utils.torch_utils import select_device
from models.experimental import attempt_load
from predict import detect
from tqdm import tqdm


def im_show(img, title='test'):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    device = select_device(args.device)

    model = attempt_load(args.weights, map_location=device)
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    for img in tqdm(os.listdir(args.images), desc='Inference', total=len(os.listdir(args.images))):
        p = os.path.join(args.images, img)
        image = cv2.imread(p)
        r_img = copy.deepcopy(image)

        tl = 3 or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)

        det = detect(image, model, device, args.size, False, 0.25, 0.45, args.classes, False)

        for i in det:
            ann = i.split(' ')
            cv2.rectangle(r_img, (int(float(ann[1])), int(float(ann[2]))),
                          (int(float(ann[3])), int(float(ann[4]))),
                          colors[int(ann[0])], 3, lineType=cv2.LINE_AA)

            c1, c2 = (int(ann[1]), int(ann[2])), (int(ann[3]), int(ann[4]))
            # label = names[int(ann[0])] + ' {}'.format(round(float(ann[-1]), 2))
            label = names[int(ann[0])]
            cv2.putText(r_img, label, (c1[0], c1[1] - 2), 0, tl / 5, colors[int(ann[0])], thickness=tf,
                        lineType=cv2.LINE_AA)

        im_show(r_img)
        # cv2.imwrite(os.path.join(args.save_dir, img), r_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./runs/train/exp2/weights/best.pt')
    parser.add_argument('--images', type=str, default='./inference')
    parser.add_argument('--save_dir', type=str, default='./result')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    main()

