#!/usr/bin/env python
# coding: utf-8
import os.path

from skimage.io import imread
import numpy as np
import glob
import io
import tqdm
import json

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
from pycocotools import mask as _mask


def _masks_as_fortran_order(masks):
    masks = masks.transpose((1, 2, 0))
    masks = np.asfortranarray(masks)
    masks = masks.astype(np.uint8)
    return masks


def _masks_as_c_order(masks):
    masks = masks.transpose((2, 0, 1))
    masks = np.ascontiguousarray(masks)
    return masks


def encode(obj):
    if len(obj.shape) == 2:
        mask = obj
        masks = np.array(np.asarray([mask]))
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        rle = rles[0]
        return rle
    elif len(obj.shape) == 3:
        masks = obj
        masks = _masks_as_fortran_order(masks)
        rles = _mask.encode(masks)
        return rles
    else:
        raise Exception("Not Implement")


def run_sam(args):
    device = "cuda"
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64, pred_iou_thresh=0.88, stability_score_thresh=0.95, points_per_batch=256)

    results = []
    paths = glob.glob(os.path.join(args.image_dir,'*.jpg'))
    paths.sort(key=lambda item: int(item.split('/')[-1].split('.')[0]))
    for path in tqdm.tqdm(paths[:]):
        img_id = int(path.split('/')[-1].split('.')[0])
        img = imread(path)
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            img = np.dstack([img] * 3)

        masks = mask_generator.generate(img)
        for mask in masks:
            score = 0.5 * mask['predicted_iou'] + 0.5 * mask['stability_score']
            seg = mask['segmentation'].astype(int)
            encodedSeg = encode(seg)
            encodedSeg['counts'] = encodedSeg['counts'].decode()
            results.append({
                'image_id': img_id,
                'category_id': 1,
                'segmentation': encodedSeg,
                'score': score
            })

    with io.open(args.output_segments, 'w', encoding='utf-8') as f:
        str_ = json.dumps(results, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        f.write(str(str_))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('image_dir', type=str,
                        help="Path to image directory, e.g., COCO's train2017 folder.")
    parser.add_argument('sam_checkpoint', type=str,
                        help='Path to SAM checkpoint based in ViT-H, like sam_vit_h_4b8939.pth')
    parser.add_argument('output_segments', type=str,
                        help='Path to outfile (.json) for generated SAM segments.')

    args = parser.parse_args()
    run_sam(args)
