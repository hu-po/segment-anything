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
        # Directory containing the dataset
        data_dir: str,
        # Number of random crops to take from fragment volume
        dataset_size: int = 16,
        # Number of points to sample per crop
        points_per_crop: int = 4,
        # Filenames of the images we'll use
        image_mask_filename='mask.png',
        image_labels_filename='inklabels.png',
        slices_dir_filename='surface_volume',
        # Expected slices per fragment
        crop_size: Tuple[int] = (3, 256, 256),
        label_size: Tuple[int] = (256, 256),
        # Depth in scan is a Clipped Normal distribution
        min_depth: int = 0,
        max_depth: int = 65,
        avg_depth: float = 27,
        std_depth: float = 10,
        # Training vs Testing mode
        train: bool = True,
        # Device to use
        device: str = 'cuda',

    ):
        print(f'Making Dataset from {data_dir}')
        self.dataset_size = dataset_size
        self.points_per_crop = points_per_crop
        self.train = train
        self.device = device
        # Open Mask image
        _image_mask_filepath = os.path.join(data_dir, image_mask_filename)
        self.mask = np.array(cv2.imread(_image_mask_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.bool)
        # Image dimmensions (depth, height, width)
        self.original_size = self.mask.shape
        self.crop_size = crop_size
        self.label_size = label_size
        # Open Label image
        if self.train:
            _image_labels_filepath = os.path.join(data_dir, image_labels_filename)
            self.labels = np.array(cv2.imread(_image_labels_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.bool)
        # Slices
        self.slice_dir = os.path.join(data_dir, slices_dir_filename)
        # Sample random crops within the image
        self.indices = np.zeros((dataset_size, 2, 3), dtype=np.int64)
        for i in range(dataset_size):
            # Select a random starting point for the subvolume
            _depth = int(np.clip(np.random.normal(avg_depth, std_depth), min_depth, max_depth))
            _height = np.random.randint(self.crop_size[1] // 2, self.original_size[0] - self.crop_size[1] // 2)
            _width = np.random.randint(self.crop_size[2] // 2, self.original_size[1] - self.crop_size[2] // 2)
            self.indices[i, 0, :] = [_depth, _height, _width]
            # End point is start point + crop size
            self.indices[i, 1, :] = [
                _depth + self.crop_size[0],
                _height + self.crop_size[1],
                _width + self.crop_size[2],
            ]

    def __len__(self):
        return self.dataset_size
    
    def _make_pixel_stats(self):
        pass

    def __getitem__(self, idx):
        # Start and End points for the crop in pixel space
        start = self.indices[idx, 0, :]
        end = self.indices[idx, 1, :]
        # Load the relevant slices and pack into image tensor
        image = torch.zeros(self.crop_size, dtype=torch.float32)
        for i, _depth in enumerate(range(start[0], end[0])):
            _slice_filepath = os.path.join(self.slice_dir, f"{_depth:02d}.tif")
            _slice = np.array(cv2.imread(_slice_filepath, cv2.IMREAD_GRAYSCALE)).astype(np.float32)
            image[i, :, :] = _slice[
                start[1] + self.crop_size[1] // 2 : end[1] - self.crop_size[1] // 2,
                start[2] + self.crop_size[2] // 2 : end[2] - self.crop_size[2] // 2,
            ]
        image = image.to(device=self.device)
        

        # Choose Points within the crop for SAM to sample
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
        dataset_size=num_samples_train,
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
        dataset_size=num_samples_valid,
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
        weights_filepath = "C:\\Users\\ook\\Documents\\dev\\segment-anything\\models\\sam_vit_b_01ec64.pth",
        model = "vit_b",
        # num_samples_train = 64,
        # device="cuda",
    )