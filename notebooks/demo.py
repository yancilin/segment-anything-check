import os
import os.path as osp

if osp.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict
from typing import Union
from segment_anything import sam_model_registry, SamPredictor


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def anno2mask(anno_list: Union[list, str], h: int, w: int):
    """ 将一个 frame 的 anno 转换成 masks """
    if isinstance(anno_list, str):
        anno_list = json.load(open(anno_list, 'r'))
    idx2polys = defaultdict(list)
    for anno in anno_list:
        idx = anno['group_id']
        poly = anno["points"]
        idx2polys[idx].append(poly)
    idx2mask = {k: polys2mask(v, h, w) for k, v in idx2polys.items()}
    return idx2mask


def polys2mask(poly_list, h, w):
    """ 异或合并多个 poly 成一个 mask """
    mask_res = np.zeros((h, w), dtype=np.uint8)
    for poly in poly_list:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, np.array([poly], dtype=np.int32), 1)
        mask_res = np.logical_xor(mask_res, mask).astype(np.uint8)
    return mask_res


def read_data(jepg_path: str, anno_path: str, show: bool = False):
    image = cv2.imread(jepg_path, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    idx2mask = anno2mask(anno_path, h, w)

    if show:
        for idx, mask in idx2mask.items():
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            show_mask(mask, plt.gca())
            plt.axis('on')
            plt.show()

    return image, idx2mask


def main():
    sam_checkpoint = '/intern-share/clyan/pretrain/sam/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    image, idx2mask = read_data(jepg_path='data/mask_test/1-1.jpg', anno_path='data/mask_test/1-1.json')
    predictor.set_image(image)
    for idx, mask_input in idx2mask.items():
        # resize mask -> (256, 256)
        mask_input = cv2.resize(mask_input, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask_input = mask_input[None, ...]
        masks, scores, logits = predictor.predict(
            mask_input=mask_input,
            multimask_output=True,
        )
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            show_mask(mask, plt.gca())
            plt.title(f"Instance: {idx + 1}, Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    main()
