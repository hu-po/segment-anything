import numpy as np
import cv2
import torch
import torchviz

from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
from segment_anything.modeling.sam import Sam

import os
from functools import partial
from typing import Tuple


# Dataset Class
class FragmentDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # Directory containing the datasets
        data_dir: str,
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
        # Expected slices per fragment
        crop: Tuple[int] = (3, 256, 256),
        # Number of subvolumes to extract from each image
        num_samples: int = 64,
        # Mean and STD for sampling slices
        mean: float = 30,
        std_dev: float = 10,
        # Min and Max values for sampling slices
        min_value: int = 0,
        max_value: int = 65,
        # Image resize ratio
        resize_ratio: float = 1.0,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = 'cuda',
        # Number of points to sample per crop
        points_per_crop: int = 20,
    ):
        print('Initializing Dataset')
        self.device = device
        # Train mode also loads the labels
        self.train = train
        self.points_per_crop = points_per_crop
        # Resize ratio reduces the size of the image
        self.resize_ratio = resize_ratio
        assert os.path.exists(
            data_dir), f"Data directory {data_dir} does not exist"
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        _mask_img = cv2.imread(_image_mask_filepath, cv2.IMREAD_GRAYSCALE)
        # Get original size and resized size
        self.height_original = _mask_img.shape[0]
        self.width_original = _mask_img.shape[1]
        self.height_resize = int(self.height_original * self.resize_ratio)
        self.width_resize = int(self.width_original * self.resize_ratio)
        self.depth_crop = crop[0]
        self.height_crop = crop[1]
        self.width_crop = crop[2]
        mask_img = cv2.resize(_mask_img, (self.width_resize, self.height_resize))
        self.mask = torch.from_numpy(np.array(mask_img)).to(dtype=torch.float32)
        self.mask = torch.nn.functional.pad(
            self.mask,
            (
                self.height_crop // 2, self.height_crop // 2,
                self.width_crop // 2, self.width_crop // 2,
            ),
            mode='constant',
            value=0,
        )
        if train:
            # Open Label image
            _image_labels_filepath = os.path.join(data_dir, image_labels_filename)
            _labels_img = cv2.imread(_image_labels_filepath, cv2.IMREAD_GRAYSCALE)
            labels_img = cv2.resize(_labels_img, (self.width_resize, self.height_resize))
            self.labels = torch.from_numpy(np.array(labels_img)).to(dtype=torch.float32)
            self.labels = torch.nn.functional.pad(
                self.labels,
                (
                    self.height_crop // 2, self.height_crop // 2,
                    self.width_crop // 2, self.width_crop // 2,
                ),
                mode='constant',
                value=0,
            )

        self.slice_dir = os.path.join(data_dir, slices_dir_filename)
        self.num_samples = num_samples
        self.indices_start = np.zeros((num_samples, 3), dtype=np.int64)
        self.indices_end = np.zeros((num_samples, 3), dtype=np.int64)
        for i in range(num_samples):

            # Select a random starting point for the subvolume
            d_start = int(np.clip(np.random.normal(mean, std_dev), min_value, max_value - 2))
            h_start = np.random.randint(self.height_resize // 2,
                                        self.height_resize - self.height_crop // 2)
            w_start = np.random.randint(self.width_resize // 2,
                                        self.width_resize - self.width_crop // 2)

            # Populate the indices matrices
            self.indices_start[i, :] = [d_start, h_start, w_start]
            self.indices_end[i, :] = [
                d_start + self.depth_crop,
                h_start + self.height_crop,
                w_start + self.width_crop,
            ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """

        Should Return:

          batched_input (list(dict)): A list over input images, each a
            dictionary with the following keys. A prompt key can be
            excluded if it is not present.
              'image': The image as a torch tensor in 3xHxW format,
                already transformed for input to the model.
              'original_size': (tuple(int, int)) The original size of
                the image before transformation, as (H, W).
              'point_coords': (torch.Tensor) Batched point prompts for
                this image, with shape BxNx2. Already transformed to the
                input frame of the model.
              'point_labels': (torch.Tensor) Batched labels for point prompts,
                with shape BxN.
              'boxes': (torch.Tensor) Batched box inputs, with shape Bx4.
                Already transformed to the input frame of the model.
              'mask_inputs': (torch.Tensor) Batched mask inputs to the model,
                in the form Bx1xHxW.
          multimask_output (bool): Whether the model should predict multiple
            disambiguating masks, or return a single mask.

        """
        start = self.indices_start[idx, :]
        end = self.indices_end[idx, :]
        crop = torch.zeros((
            self.depth_crop,
            self.height_crop,
            self.width_crop,
        ), dtype=torch.float32)
        for i, slice in enumerate(range(start[0], end[0])):
            slice_filepath = os.path.join(self.slice_dir, f"{slice:02d}.tif")
            cv2_img = cv2.imread(slice_filepath, cv2.IMREAD_GRAYSCALE)
            cv2_img = cv2.resize(cv2_img, (self.width_resize, self.height_resize))
            cv2_img = torch.from_numpy(np.array(cv2_img)).to(dtype=torch.float32)
            cv2_img = torch.nn.functional.pad(
                cv2_img,
                (
                    self.height_crop // 2, self.height_crop // 2,
                    self.width_crop // 2, self.width_crop // 2,
                ),
                mode='constant',
                value=0,
            )
            crop[i, :, :] = cv2_img[start[1]:end[1], start[2]:end[2]]
        
        # Choose N random points within the crop
        self.points_per_crop = 10
        point_coords = torch.zeros((self.points_per_crop, 2), dtype=torch.long)
        point_labels = torch.zeros(self.points_per_crop, dtype=torch.long)
        for i in range(self.points_per_crop):
            point_coords[i, 0] = np.random.randint(0, self.height_crop)
            point_coords[i, 1] = np.random.randint(0, self.width_crop)
            point_labels[i] = self.labels[
                start[1] + point_coords[i, 0],
                start[2] + point_coords[i, 1],
            ]

        return {
            'image': crop.to(device=self.device),
            'point_coords': point_coords.unsqueeze(0).to(device=device),
            'point_labels': point_labels.unsqueeze(0).to(device=device),
            'original_size': (self.height_crop, self.width_crop),
            'crop_dims': (start, end)
        }


# Train, Valid DataLoader
batch_size = 2
crop = (3, 224, 224)
num_samples_train = 64
num_samples_valid = 64
resize_ratio = 1.0
train_dataset = FragmentDataset(
    data_dir="/home/tren/dev/ashenvenus/data/split_train/1",
    num_samples=num_samples_train,
    crop=crop,
    resize_ratio=resize_ratio,
    train=True,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    collate_fn=lambda x: x,
    batch_size=batch_size,
    shuffle=True,
    # pin_memory=True,
)
valid_dataset = FragmentDataset(
    data_dir="/home/tren/dev/ashenvenus/data/split_valid/1",
    num_samples=num_samples_valid,
    crop=crop,
    resize_ratio=resize_ratio,
    train=True,
)
valid_loader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    collate_fn=lambda x: x,
    batch_size=batch_size,
    shuffle=False,
    # pin_memory=True,
)

# Model
checkpoint = "/home/tren/dev/segment-anything/models/sam_vit_b_01ec64.pth"
device = "cuda:0"
encoder_embed_dim = 768
encoder_depth = 12
encoder_num_heads = 12
encoder_global_attn_indexes = [2, 5, 8, 11]
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size
print("Creating Sam model")
sam = Sam(
    image_encoder=ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    ),
    prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    ),
    mask_decoder=MaskDecoder(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ),
    # TODO: Get from Dataset
    pixel_mean=[123.675, 116.28, 103.53],
    pixel_std=[58.395, 57.12, 57.375],
)
if checkpoint is not None:
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    sam.load_state_dict(state_dict)
sam = sam.to(device=device)
sam.train()
print('\n\n\n TRAINABLE PARAMETERS \n\n\n')
for name, param in sam.named_parameters():
    if param.requires_grad:
        print(f"{name} : {param.shape}")

# Optimizer
lr = 1e-4
wd = 1e-4
optimizer = torch.optim.Adam(sam.parameters(), lr=lr, weight_decay=wd)

# Loss
loss_fn = torch.nn.CrossEntropyLoss()

# Training
num_epochs = 2
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")

    for i, batch in enumerate(train_loader):
        print(f"Batch {i}")

        print(f" batch[0]['image'] {batch[0]['image'].shape}")
        print(f" batch[0]['image'] {batch[0]['image'].dtype}")
        # print(f" batch[0]['mask_inputs'] {batch[0]['mask_inputs'].shape}")
        # print(f" batch[0]['mask_inputs'] {batch[0]['mask_inputs'].dtype}")
        print(f" batch[0]['original_size'] {batch[0]['original_size']}")
        start = batch[0]['crop_dims'][0]
        end = batch[0]['crop_dims'][1]

        output = sam(batch, multimask_output=False)
        """
        
        Should Return:

            (list(dict)): A list over input images, where each element is
            as dictionary with the following keys.
              'masks': (torch.Tensor) Batched binary mask predictions,
                with shape BxCxHxW, where B is the number of input promts,
                C is determiend by multimask_output, and (H, W) is the
                original size of the image.
              'iou_predictions': (torch.Tensor) The model's predictions
                of mask quality, in shape BxC.
              'low_res_logits': (torch.Tensor) Low resolution logits with
                shape BxCxHxW, where H=W=256. Can be passed as mask input
                to subsequent iterations of prediction.

        """
        for i, out in enumerate(output):
            gt_mask = train_dataset.mask[start[1]:end[1], start[2]:end[2]]
            gt_mask = gt_mask.unsqueeze(0).unsqueeze(0).to(device=device)
            pred_mask = out['masks'].to(dtype=torch.float32)
            print(f"Pred Mask Shape: {pred_mask.shape}")
            print(f"Pred Mask Type: {pred_mask.dtype}")
            print(f"Pred Mask Max: {pred_mask.max()}")
            print(f"Pred Mask Min: {pred_mask.min()}")
            print(f"GT Mask Shape: {gt_mask.shape}")
            print(f"GT Mask Type: {gt_mask.dtype}")
            print(f"GT Mask Max: {gt_mask.max()}")
            print(f"GT Mask Min: {gt_mask.min()}")
            loss = loss_fn(pred_mask, gt_mask)
            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
