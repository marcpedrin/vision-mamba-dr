"""
Fundus preprocessing and augmentation.

    crop black border -> pad to square -> resize 128 -> (train: flips, rotation, jitter)
    -> tensor -> normalise.  PIL image any size -> (3, 128, 128)

    No random crop / erasing / mixup like the paper, they can remove the tiny
    lesions that separate grade 1 from grade 2. No hue jitter, lesion colour matters.

"""

import numpy as np
from PIL import Image
from torchvision import transforms

#standard ImageNet RGB mean/std (no pretrained weights, just need the same for train and test)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CropToRetina:
    #Crop away the black frame around the round retina.

    def __init__(self, threshold=10):
        #above this grey value counts as retina (jpeg noise in the black)
        self.threshold = threshold

    def __call__(self, image):
        #grey (H, W) 0-255
        grey = np.asarray(image.convert("L"))

        mask = grey > self.threshold

        #rows / cols with any retina pixel
        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]

        #all black image, leave it
        if len(rows) == 0 or len(cols) == 0:
            return image

        #bounding box, right/bottom are exclusive so +1
        left = int(cols[0])
        top = int(rows[0])
        right = int(cols[-1]) + 1
        bottom = int(rows[-1]) + 1
        return image.crop((left, top, right, bottom))


class PadToSquare:
    #Pad with black to a square so resizing doesnt squash the retina.

    def __call__(self, image):
        width, height = image.size
        side = max(width, height)
        canvas = Image.new(image.mode, (side, side), 0)
        #paste in the middle
        left = (side - width) // 2
        top = (side - height) // 2
        canvas.paste(image, (left, top))
        return canvas


def build_train_transform(img_size=128, crop_threshold=10, rotation_degrees=15.0,
                          color_jitter=0.1):
    """Training pipeline: preprocessing + random augmentation."""
    return transforms.Compose(
        [
            CropToRetina(crop_threshold),
            PadToSquare(),
            #resize first so the random steps are cheap
            transforms.Resize((img_size, img_size)),
            #grade doesnt depend on left/right or up/down
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            #+-15 degrees, head tilt
            transforms.RandomRotation(degrees=rotation_degrees),
            #up to 10%, no hue
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            ),
            #0-255 -> (3, 128, 128) in [0, 1]
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(img_size=128, crop_threshold=10):
    """Val/test pipeline: same preprocessing, no randomness."""
    return transforms.Compose(
        [
            CropToRetina(crop_threshold),
            PadToSquare(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
