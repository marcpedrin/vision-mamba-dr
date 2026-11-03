"""
Messidor-2 dataset: external test set, EVALUATION ONLY, never trained on.

    1,748 images, adjudicated grades 0-4 (Krause 2018).
    Ungradable images have no grade and are skipped.
    Column names come from the config.

"""

import os
import warnings

import pandas as pd
from torch.utils.data import Dataset

#same loader as EyePACS
from .eyepacs import load_rgb_image


def resolve_image_path(image_dir, image_id, default_ext):
    """Path for one id, adds default_ext only if the id has no extension."""
    _, ext = os.path.splitext(image_id)
    if not ext:
        image_id = image_id + default_ext
    return os.path.join(image_dir, image_id)


class Messidor2Dataset(Dataset):
    #Messidor-2 (image_tensor, label) pairs. EVAL ONLY.

    def __init__(self, image_dir, labels_csv, transform=None,
                 id_column="image_id", label_column="adjudicated_dr_grade",
                 default_ext=".png", draft_size=512):
        super().__init__()

        self.transform = transform
        self.draft_size = draft_size

        table = pd.read_csv(labels_csv)

        for column in (id_column, label_column):
            if column not in table.columns:
                raise ValueError(
                    f"{labels_csv} has no '{column}' column; set "
                    f"data.messidor_id_column / data.messidor_label_column in the "
                    f"config to the names your copy uses"
                )

        #ungradable = empty grade (NaN), drop them
        gradable = table[table[label_column].notna()]
        skipped_ungradable = len(table) - len(gradable)

        self.samples = []
        missing = []
        for image_id, grade in zip(gradable[id_column], gradable[label_column]):
            path = resolve_image_path(image_dir, str(image_id), default_ext)
            if os.path.exists(path):
                self.samples.append((path, int(grade)))
            else:
                missing.append(path)

        if skipped_ungradable:
            print(f"Messidor2Dataset: skipped {skipped_ungradable} ungradable image(s)")
        if missing:
            warnings.warn(
                f"Messidor2Dataset: {len(missing)} image(s) listed in the CSV were "
                f"not found and will be skipped (e.g. {missing[0]})"
            )

        self.labels = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        try:
            image = load_rgb_image(path, self.draft_size)
        except OSError as error:
            raise RuntimeError(f"Could not read image {path}") from error

        #same eval preprocessing as EyePACS test
        if self.transform is not None:
            image = self.transform(image)

        return image, label
