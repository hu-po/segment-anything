import numpy as np
import cv2
import torch

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
        # Number of subvolumes to extract from each image
        num_samples: int = 64,
        # Mean and STD for sampling slices
        mean: float = 30,
        std_dev: float = 10,
        # Min and Max values for sampling slices
        min_value: int = 0,
        max_value: int = 65,
        # Expected slices per fragment
        crop: Tuple[int] = (3, 224, 224),
        # Image resize ratio
        resize_ratio: float = 1.0,
        # Training vs Testing mode
        train: bool = True,
    ):
        # Train mode also loads the labels
        self.train = train
        # Resize ratio reduces the size of the image
        self.resize_ratio = resize_ratio
        assert os.path.exists(
            data_dir), f"Data directory {data_dir} does not exist"
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        _mask_img = cv2.imread(_image_mask_filepath, cv2.IMREAD_GRAYSCALE)
        # Get original size and resized size
        self.original_size = _mask_img.size
        self.resize_height = int(self.original_size[0] * self.resize_ratio)
        self.resize_width = int(self.original_size[1] * self.resize_ratio)
        mask_img = cv2.resize(_mask_img, (self.resize_height, self.resize_width))
        self.mask = torch.from_numpy(np.array(mask_img)).to(torch.bool)
        if train:
            # Open Label image
            _image_labels_filepath = os.path.join(data_dir, image_labels_filename)
            _labels_img = cv2.imread(_image_labels_filepath, cv2.IMREAD_GRAYSCALE)
            labels_img = cv2.resize(_labels_img, (self.resize_height, self.resize_width))
            self.labels = torch.from_numpy(np.array(labels_img)).to(torch.bool)

        self.slice_dir = os.path.join(data_dir, slices_dir_filename)

        self.num_samples = num_samples
        self.indices_start = np.zeros[num_samples, 3]
        self.indices_end = np.zeros[num_samples, 3]
        for i in range(num_samples):

            # Select a random number using a Normal distribution with the specified mean and standard deviation
            start_float = np.random.normal(mean, std_dev)
            # Make sure the starting number is within the valid range and convert it to an integer
            z = int(np.clip(start_float, min_value, max_value - 2))

            # Select a random starting point for the subvolume
            x = np.random.randint(0, self.resize_width - crop[2])
            y = np.random.randint(0, self.resize_height - crop[1])

            # Populate the indices matrices
            self.indices_start[i, :] = [z, x, y]
            self.indices_end[i, :] = [z + crop[0], x + crop[2], y + crop[1]]

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

        img = np.zeros((self.resize_width, self.resize_height, 3), dtype=np.uint8)    
        for i, slice in enumerate(range(start[0], end[0])):
            slice_filepath = os.path.join(self.slice_dir, f"{slice:02d}.tif")
            cv2_img = cv2.imread(slice_filepath, cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(cv2_img, (self.resize_width, self.resize_height))
            img[:, :, i] = resized_img[:, :]

        _dict = {
            'image': img,
            'original_size': (self.resize_width, self.resize_height),
            'mask_inputs': None,
        }
        return

# Train, Valid DataLoader
batch_size = 2
crop = (3, 224, 224)
num_samples_train = 64
num_samples_valid = 64
resize_ratio = 1.0
train_loader = torch.utils.data.DataLoader(
    dataset=FragmentDataset(
        data_dir="/home/tren/dev/ashenvenus/data/split_train/1",
        num_samples=num_samples_train,
        crop=crop,
        resize_ratio=resize_ratio,
        train=True,
    ),
    batch_size=batch_size,
    shuffle=True,
    pin_memory=True,
)
valid_loader = torch.utils.data.DataLoader(
    dataset=FragmentDataset(
        data_dir="/home/tren/dev/ashenvenus/data/split_valid/1",
        num_samples=num_samples_valid,
        crop=crop,
        resize_ratio=resize_ratio,
        train=True,
    ),
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
)

# Model
checkpoint = "/home/tren/dev/segment-anything/models/sam_vit_b_01ec64.pth"
device = "cuda:0"
encoder_embed_dim=768
encoder_depth=12
encoder_num_heads=12
encoder_global_attn_indexes=[2, 5, 8, 11]
prompt_embed_dim = 256
image_size = 1024
vit_patch_size = 16
image_embedding_size = image_size // vit_patch_size
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
    # pixel_mean=[123.675, 116.28, 103.53],
    # pixel_std=[58.395, 57.12, 57.375],
)
sam.train()
if checkpoint is not None:
    with open(checkpoint, "rb") as f:
        state_dict = torch.load(f)
    sam.load_state_dict(state_dict)
sam = sam.to(device=device)

# Optimizer
lr = 1e-4
wd = 1e-4
optimizer = torch.optim.Adam(sam.parameters(), lr=lr, weight_decay=wd)

# Loss
loss_fn = torch.nn.CrossEntropyLoss()

# Training
num_epochs = 0
for epoch in range(num_epochs):

    for batch in train_loader:
        # Forward
        output = sam.forward(batch, multimask_output=False)
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

        output['masks']
        output['iou_predictions']
        output['low_res_logits']

        # Backward
        loss = loss_fn(output['masks'], batch['mask_inputs'])

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    pass
