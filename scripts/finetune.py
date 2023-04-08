import numpy as np
import cv2
import torch
from tqdm import tqdm

from segment_anything.modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)
from segment_anything.modeling.sam import Sam
from segment_anything import sam_model_registry
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader

import os
from typing import Tuple


# Dataset Class
class FragmentDataset(Dataset):
    def __init__(
        self,
        # Directory containing the datasets
        data_dir: str,
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
        # Expected slices per fragment
        crop_size: Tuple[int] = (3, 256, 256),
        label_size: Tuple[int] = (256, 256),
        # Number of subvolumes to extract from each image
        num_samples: int = 2,
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
        self.depth_crop = crop_size[0]
        self.height_crop = crop_size[1]
        self.width_crop = crop_size[2]
        self.label_size = label_size
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
        if self.train:
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
        image = crop.to(device=self.device)

        # Choose N random points within the crop
        point_coords = torch.zeros((self.points_per_crop, 2), dtype=torch.long)
        point_labels = torch.zeros(self.points_per_crop, dtype=torch.long)
        for i in range(self.points_per_crop):
            point_coords[i, 0] = np.random.randint(0, self.height_crop)
            point_coords[i, 1] = np.random.randint(0, self.width_crop)
            point_labels[i] = self.labels[
                start[1] + point_coords[i, 0],
                start[2] + point_coords[i, 1],
            ]
        point_coords = point_coords.to(device=self.device)
        point_labels = point_labels.to(device=self.device)
        if self.train:
            labels = self.labels[
                    start[1]:end[1],
                    start[2]:end[2],
            ]
            # convert to cv2 image
            labels = labels.numpy()
            labels = cv2.resize(labels, self.label_size)
            labels = torch.from_numpy(labels).to(dtype=torch.float32)
            labels = labels.unsqueeze(0).clone().to(device=self.device)
            return image, point_coords, point_labels, labels
        else:
            return image, point_coords, point_labels

def train_valid(
    output_dir: str = "/home/tren/dev/segment-anything/output/train",
    train_dir: str = "/home/tren/dev/ashenvenus/data/split_train/1",
    valid_dir: str = "/home/tren/dev/ashenvenus/data/split_valid/1",
    model: str = "vit_b",
    weights_filepath: str = "/home/tren/dev/segment-anything/models/sam_vit_b_01ec64.pth",
    num_samples_train: int = 2,
    num_samples_valid: int = 2,
    batch_size: int = 1,
    optimizer: str = "adam",
    lr: float = 1e-4,
    wd: float = 1e-4,
    image_augs: bool = False,
    crop_size: Tuple[int] = (3, 1024, 1024),
    resize_ratio: float = 1.0,
    num_epochs: int = 2,
    save_model: bool = True,
    device: str = "cpu",  # "cuda:0"
    **kwargs,
):
    train_dataset = FragmentDataset(
        data_dir=train_dir,
        num_samples=num_samples_train,
        crop_size=crop_size,
        resize_ratio=resize_ratio,
        train=True,
        device=device,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        # pin_memory=True,
    )
    valid_dataset = FragmentDataset(
        data_dir=valid_dir,
        num_samples=num_samples_valid,
        crop_size=crop_size,
        resize_ratio=resize_ratio,
        train=True,
        device=device,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        # pin_memory=True,
    )

    model = sam_model_registry[model](checkpoint=weights_filepath)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=output_dir)

    step = 0
    score = 0    
    best_score = 0
    for epoch in range(num_epochs):
        print(f"\n\n --- Epoch {epoch} --- \n\n")

        print("Training...")
        score = 0
        _loader = tqdm(train_loader)
        for images, point_coords, point_labels, labels in _loader:
            writer.add_images("input.image/train", images, step)
            writer.add_images("input.label/train", labels, step)
            # # Plot point coordinates into a blank image of size images
            # point_coords = point_coords.cpu().numpy()
            # point_labels = point_labels.cpu().numpy()
            # point_image = np.zeros(images.shape)
            # for i in range(point_coords.shape[0]):
            #     point_image[0, point_coords[i, 0], point_coords[i, 1]] = point_labels[i]
            # writer.add_images("Train.Points", point_image, step)
            image_embeddings = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            # HACK: Something goes on here for batch sizes greater than 1
            # TODO: iou predictions could be used for additional loss
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            writer.add_images("output.masks/train", low_res_masks, step)
            loss = loss_fn(low_res_masks, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

            _loss_name = f"{loss_fn.__class__.__name__}/train"
            writer.add_scalar(f"{_loss_name}", loss.item(), step)
            _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
        
        print("Validating...")
        score = 0
        _loader = tqdm(valid_loader)
        for images, point_coords, point_labels, labels in _loader:
            writer.add_images("input.image/valid", images, step)
            writer.add_images("input.label/valid", labels, step)
            image_embeddings = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            writer.add_images("output.masks/valid", low_res_masks, step)
            loss = loss_fn(low_res_masks, labels)
            score -= loss.item()

            _loss_name = f"{loss_fn.__class__.__name__}/valid"
            writer.add_scalar(f"{_loss_name}", loss.item(), step)
            _loader.set_postfix_str(f"{_loss_name}: {loss.item():.4f}")
        
        score /= len(valid_loader)
        if score > best_score:
            print(f"New best score! >> {score:.4f} (was {best_score:.4f})")        
            best_score = score
            if save_model:
                _model_filepath = os.path.join(output_dir, f"model_{epoch}.pth")
                print(f"Saving model to {_model_filepath}")
                torch.save(model.state_dict(), _model_filepath)

        # Flush writer
        writer.flush()
    writer.close()

    return score

if __name__ == "__main__":
    
        
    train_valid(
        train_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split_train\\1",
        valid_dir = "C:\\Users\\ook\\Documents\\dev\\ashenvenus\\data\\split_valid\\1",
        output_dir = "C:\\Users\\ook\\Documents\\dev\\segment-anything\\output\\",
        model = "vit_b",
        weights_filepath = "C:\\Users\\ook\\Documents\\dev\\segment-anything\\models\\sam_vit_b_01ec64.pth",
        # num_samples_train = 64,
        # device="cuda",
    )