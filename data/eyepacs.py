"""
EyePACS dataset (Kaggle 2015), the training data.

    trainLabels.csv: image,level  e.g. 10_left,0
    grades 0 no DR, 1 mild, 2 moderate, 3 severe, 4 proliferative (~73% are 0)
    one item: (3, 128, 128) tensor, int label

"""

import os
import warnings

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def read_eyepacs_labels(csv_path):
    """Read trainLabels.csv and add a patient_id column ("10_left" -> "10")."""
    table = pd.read_csv(csv_path, dtype={"image": str})

    for column in ("image", "level"):
        if column not in table.columns:
            raise ValueError(f"{csv_path} has no '{column}' column")

    #split at the last _ -> patient id
    table["patient_id"] = table["image"].str.rsplit("_", n=1).str[0]

    table["level"] = table["level"].astype(int)

    return table


def load_rgb_image(path, draft_size=None):
    """Open an image as RGB. draft_size lets jpeg decode smaller (much faster)."""
    with Image.open(path) as image:
        if draft_size is not None:
            image.draft("RGB", (draft_size, draft_size))
        #force 3 channels, also loads pixels before the file closes
        return image.convert("RGB")


class EyePACSDataset(Dataset):
    #(image_tensor, label) pairs, images read from disk only when asked for

    def __init__(self, image_dir, labels, transform=None, image_ext=".jpeg",
                 draft_size=512):
        super().__init__()

        self.transform = transform
        self.draft_size = draft_size

        #list of (path, label), skip files that arent on disk
        self.samples = []
        missing = []
        for name, level in zip(labels["image"], labels["level"]):
            #"10_left" -> image_dir/10_left.jpeg
            path = os.path.join(image_dir, name + image_ext)
            if os.path.exists(path):
                self.samples.append((path, int(level)))
            else:
                missing.append(path)

        #one warning instead of crashing mid epoch
        if missing:
            warnings.warn(
                f"EyePACSDataset: {len(missing)} image(s) listed in the labels were "
                f"not found and will be skipped (e.g. {missing[0]})"
            )

        #labels only, for class weights without opening images
        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        #corrupt file -> error with the file name
        try:
            image = load_rgb_image(path, self.draft_size)
        except OSError as error:
            raise RuntimeError(f"Could not read image {path}") from error

        #-> (3, 128, 128)
        if self.transform is not None:
            image = self.transform(image)

        return image, label
