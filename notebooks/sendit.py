from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    box_xyxy_to_xywh,
    rle_to_mask,
)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import gc
import os
import pprint
import shutil
import sys
from hyperopt import fmin, hp

sys.path.append("..")

output_dir = '/home/tren/dev/segment-anything/notebooks/images/masks'
sam_checkpoint = "/home/tren/dev/segment-anything/models/sam_vit_b_01ec64.pth"
device = "cuda:0"
model_type = "vit_b"
mean = 25
std_dev = 10
min_value, max_value = 0, 65


def clear_gpu_memory():
    if torch.cuda.is_available():
        print('Clearing GPU memory')
        torch.cuda.empty_cache()
        gc.collect()


clear_gpu_memory()
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:, :, i] = color_mask[i]
        ax.imshow(np.dstack((img, m * 0.35)))


def make_mask(hparams):

    print(f"Hyperparameters: {pprint.pformat(hparams)}")

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=hparams['points_per_side'],
        pred_iou_thresh=hparams['pred_iou_thresh'],
        stability_score_thresh=hparams['stability_score_thresh'],
        crop_n_layers=hparams['crop_n_layers'],
        crop_n_points_downscale_factor=hparams['crop_n_points_downscale_factor'],
        # min_mask_region_area=hparams['min_mask_region_area'],
    )

    mask_img = cv2.imread('/home/tren/dev/ashenvenus/data/train/1/inklabels.png', cv2.IMREAD_GRAYSCALE)
    img_width = int(mask_img.shape[0] * hparams['resize'])
    img_height = int(mask_img.shape[1] * hparams['resize'])
    resized_mask_img = cv2.resize(mask_img, (img_height, img_width))

    slice_dir = '/home/tren/dev/ashenvenus/data/train/1/surface_volume/'
    
    mask_data: MaskData = None
    for j in range(hparams['num_samples']):

        # Select a random number using a Normal distribution with the specified mean and standard deviation
        start_float = np.random.normal(mean, std_dev)
        # Make sure the starting number is within the valid range and convert it to an integer
        start = int(np.clip(start_float, min_value, max_value - 2))

        img = np.zeros((img_width, img_height, 3), dtype=np.uint8)    
        for i, slice in enumerate([start, start + 1, start + 2]):
            print(f"Reading slice {slice}...")
            slice_filepath = os.path.join(slice_dir, f"{slice:02d}.tif")
            cv2_img = cv2.imread(slice_filepath, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(cv2_img, (img_height, img_width))
            img[:, :, i] = resized_img[:, :]

        # masks = mask_generator.generate(img)
        _mask_data: MaskData = mask_generator._generate_masks(img)
        # import pdb; pdb.set_trace()
        if j == 0:
            mask_data = _mask_data
        else:
            mask_data.cat(_mask_data)

        # Filter small disconnected regions and holes in masks
    if mask_generator.min_mask_region_area > 0:
        mask_data = mask_generator.postprocess_small_regions(
            mask_data,
            mask_generator.min_mask_region_area,
            max(mask_generator.box_nms_thresh, mask_generator.crop_nms_thresh),
        )

    mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
        }
        curr_anns.append(ann)
        
    plt.figure(figsize=(20, 20))
    plt.imshow(resized_mask_img)
    show_anns(curr_anns)
    plt.axis('off')
    # plt.show()

    # Concatenate the hyperparameters into a string
    hparams_str = "_".join([f"{k}_{v}" for k, v in hparams.items()])

    # Save image to output dir
    output_filepath = os.path.join(output_dir, f"{hparams_str}.png")
    plt.savefig(output_filepath)

    return 0


if __name__ == "__main__":

    search_space = {
        "num_samples": 3,
        "resize": hp.choice("resize", [0.1]),
        "points_per_side": hp.choice("points_per_side", [8, 16, 32]),
        "pred_iou_thresh": hp.choice("pred_iou_thresh", [0.5, 0.7, 0.86]),
        "stability_score_thresh": hp.choice("stability_score_thresh", [0.5, 0.7, 0.8, 0.92]),
        "crop_n_layers" : hp.choice("crop_n_layers", [0, 1, 2]),
        "crop_n_points_downscale_factor" : hp.choice("crop_n_points_downscale_factor", [1, 2]),
        "box_nms_thresh" : hp.choice("box_nms_thresh", [0.5, 0.7, 0.8]),
        "crop_nms_thresh" : hp.choice("crop_nms_thresh", [0.5, 0.7, 0.8]),
        "min_mask_region_area": 100,  # Requires open-cv to run post-processing
    }

    # Clean output dir
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir)

    best = fmin(
        make_mask,
        space=search_space,
        max_evals=100,
        rstate=np.random.Generator(np.random.PCG64(42)),
    )
