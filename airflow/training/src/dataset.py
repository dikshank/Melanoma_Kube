# This file is supposed to return us the dataset creation logic, as we know training data requires different on the go processing, and val or test requires different kind of preprocessing, keep in mind that this is different from the standard preprocessing that we do for the whole data combined, that whole preproessing is also automated by the airflow dags, in airflow the script can run over and over again as we have new raw data incoming so we don't have to stress about that, and yes the scripts stays the same and usually we have config files to let the script know which data we are going to preprocess, keep in mind that now matter how much data we have the preprocessing logic in airflow is based on a predefined script so all data is going to processed by that logic so there are no inconsistencies in the processed data, if you wish to create a new logic, the it is better to do it in a new folder using a new airflow dag
# The airflow propreprocessing is this below":0
# Airflow preprocessing
    # ↓
# Creates/prepares the dataset
    # ↓
# processed images + CSV manifests



# JUSTICATION FOR THE CURRENT APPROACH:

# I separated offline preprocessing from runtime dataset transformations because they serve different purposes. The Airflow pipeline handles deterministic, dataset-level processing such as validation, splitting, and the preprocessing that has already been applied to the melanoma images. That pipeline is reproducible and can be rerun whenever new raw data arrives.

# The PyTorch Dataset handles transformations that need to happen dynamically when samples are loaded. Training and evaluation have different requirements, so I apply augmentation only to the training split and keep validation and test deterministic.

# For training, I use flips, rotations, and moderate ColorJitter. The goal isn't to remove medically relevant color information; it's to make the model less sensitive to small changes in illumination, camera characteristics, and acquisition conditions while preserving the underlying lesion characteristics.

# I deliberately don't use aggressive artifact removal such as hair removal in this Dataset pipeline. Since the eventual goal is generalization to real-world images, I don't want to artificially make the training distribution cleaner than the deployment distribution unless I have evidence that the artifact-removal procedure improves robustness.

# Finally, I use ImageNet normalization because the model uses pretrained weights. The normalization keeps the input distribution consistent with the preprocessing expected by those pretrained weights.

# NOTE THAT, Augmentation (usually) should not be a part of the actual storage because it will increase the disk size basically augmentation is stochastic and training-specific. I don't want to permanently store one randomly augmented version of an image. By applying it in the Dataset, the same original image can receive different augmentations across epochs, while validation and test always use deterministic transforms


# Now in this file let's create the logic

import pandas as pd # because we have csv files that have path
from pathlib import Path # because csv files have paths
import torch 
from torch.utils.data import Dataset # because this is the goal, so that in training, the dataloader can take the Dataset object as input
from torchvision import transforms # because we have to transform the images
from PIL import Image # to return the actual numerical image from path

# while creating the labels in the proprocessing I have just used the names so I have this mapping here to convert them to numbers
LABEL_MAP = {
    "benign": 0,
    "malignant": 1
}

def get_transform(img_size, split):
    """
    Return the appropriate transforms for a given split.

    Training split gets augmentation transforms.
    Val and test splits get only resize + normalize (no augmentation).

    Augmentation is applied here (not in the pipeline) to avoid
    storing augmented images on disk. Basically doing it on the fly

    Args:
        img_size (int): Target image size (square). E.g. 32 or 64.
        split (str): One of 'train', 'val', 'test'.

    Returns:
        transforms.Compose object: Composed transform pipeline.
    """

    # ImageNet normalization stats (used since we have pretrained weights)
    # Many commonly used pretrained vision models were pretrained on ImageNet, so their weights were learned with ImageNet-style input preprocessing. Using the corresponding ImageNet normalization keeps the inputs in a distribution compatible with what the pretrained model saw during pretraining.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        # ── Heavy augmentation for training ──────────────────────────────
        # Critical for small datasets (600 images) to prevent overfitting
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    else:
        # ── No augmentation for val and test ─────────────────────────────
        # Only resize and normalize — deterministic, reproducible
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


class MelanomaDataset(Dataset):
    '''
    By using Dataset in the braces, it is a way to inherit the functions of the pytorch's Dataset class
    
    Reads image paths and labels from a CSV manifest file produced by the Airflow split pipeline.
    
    input args:
        manifest_path (str or Path): Path to split manifest CSV. (Columns: filepath, label)
        img_size (int): Target image size (square). # we can choose to upscale of downscale images based on the input.
        split (str): One of 'train', 'val', 'test'.  # Since the manifest exist for all 3, while creating the Dataset object later in training we have to define for which phase are we defining this for.
        debug (bool): If True, loads only debug_size images.  # In the beginning of any project when we are ready to run the training pipeline, it is never a good idea to use full dataset for all the epochs, so we have debug mode where we will take only a few data points / images initially, this is a good practice to keep in mind 
        debug_size (int): Number of images to load in debug mode. # This is for how many datapoints/images we want to keep for the debug mode.
    '''

    def __init__(self, manifest_path: str, img_size: int, split: str, debug: bool = False, debug_size: int = 20):
        
        self.split = split
        self.img_size = img_size
        self.debug = debug
        self.transform = get_transform(img_size, split)  # this is the tranform function which will return an "object"

        # ── Load manifest ─────────────────────────────────────────────────
        manifest_path = Path(manifest_path) # This converts manifest_path into a Path object from Python's pathlib module. Because Path gives you convenient filesystem operations like Path.exists() and so on

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run the Airflow pipeline first."
            )

        df = pd.read_csv(manifest_path)

        # ── Validate manifest columns ─────────────────────────────────────
        if "filepath" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"Manifest must have 'filepath' and 'label' columns. "
                f"Found: {df.columns.tolist()}"
            )

        # ── Debug mode: subsample ─────────────────────────────────────────
        if debug:
            # Take equal samples from each class if possible
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda x: x.sample(
                    min(len(x), debug_size // 2),
                    random_state=42
                ))
                .reset_index(drop=True)
            )

        self.filepaths = df["filepath"].tolist()
        self.labels = [LABEL_MAP[label] for label in df["label"].tolist()]


    # A Dataset object must have two things, one is __len__ and one __getitem__ for the dataloader to call (dataloader will have __iter__ to call the __getitem__, like for i in range(), range also has __iter__ in built

    def __len__(self):
        return len(self.filepaths)
    

    def __getitem__(self, idx: int):
        """
        Load and return a single image and its label.

        Args:
            idx (int): Index of the sample which the dataloader will pass while in dataloader.__iter__().

        Returns:
            tuple: (image_tensor, label) where label is 0=benign, 1=malignant.
        """
        img_path = self.filepaths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)   # using the same transform object that we create in init using the get_transformer fuction
        return img_tensor, label
