#!/usr/bin/env python
# coding: utf-8
import os.path

from skimage.io import imread
import numpy as np
import glob
import io
import tqdm
import json

from segment_anything import SamPredictor, sam_model_registry

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

    mask_generator = SamPredictor(sam)

    with open(args.prompts) as data_file:
        prompts = json.loads(data_file.read())

    results = []
    paths = glob.glob(os.path.join(args.image_dir,'*.jpg'))
    paths.sort(key=lambda item: int(item.split('/')[-1].split('.')[0]))
    for path in tqdm.tqdm(paths[:]):
        img_id = int(path.split('/')[-1].split('.')[0])
        img = imread(path)
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            img = np.dstack([img] * 3)

        if not str(img_id) in prompts:
            continue
        else:
            mask_generator.set_image(img)
            for center in prompts[str(img_id)]:
                centerArray = np.array(center[:2]).reshape(1, -1)
                centerArray[0][0] *= w
                centerArray[0][1] *= h
                cls = center[-1]
                masks, scores, _ = mask_generator.predict(point_coords=centerArray, point_labels=[True],
                                                          multimask_output=True)
                for i in range(len(masks)):
                    seg = masks[i].astype(int)
                    score = float(scores[i])
                    encodedSeg = encode(seg)
                    encodedSeg['counts'] = encodedSeg['counts'].decode()
                    results.append({
                        'image_id': img_id,
                        'category_id': 1,
                        'segmentation': encodedSeg,
                        'score': score,
                        'clsID': cls,
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
    parser.add_argument('prompts', type=str,
                        help='Path to json prompt file generated with one of the given scripts in the main git of SOS.')
    parser.add_argument('output_segments', type=str,
                        help='Path to outfile (.json) for generated SAM segments.')

    args = parser.parse_args()
    run_sam(args)
