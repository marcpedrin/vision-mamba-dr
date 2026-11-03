"""
Fake fundus images for the tests and --smoke-test (never used for results).

    Orange disc on black with dark red dots, more dots = higher grade.
    Writes CSVs in the same format as EyePACS and Messidor-2.

"""

import os

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def _draw_fake_fundus(grade, rng, width=200, height=160):
    """One fake fundus image, wider than tall so crop/pad gets tested."""
    image = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    #the "retina"
    radius = 70
    cx, cy = width // 2, height // 2
    red = 180 + rng.randint(0, 40)
    green = 80 + rng.randint(0, 30)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                 fill=(red, green, 40))

    #"lesions": 6 dots per grade inside the disc
    for _ in range(6 * grade):
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(0, 0.8 * radius)
        x = cx + dist * np.cos(angle)
        y = cy + dist * np.sin(angle)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(90, 10, 10))

    return image


def make_synthetic_eyepacs(folder, num_patients=10, seed=0):
    """Left + right eye per patient and trainLabels.csv. Returns (image_dir, labels_csv)."""
    rng = np.random.RandomState(seed)

    image_dir = os.path.join(folder, "train")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for patient in range(num_patients):
        patient_id = patient + 1
        #cycle grades so every grade shows up
        grade = patient % 5
        for eye in ("left", "right"):
            name = f"{patient_id}_{eye}"
            image = _draw_fake_fundus(grade, rng)
            image.save(os.path.join(image_dir, name + ".jpeg"), quality=90)
            rows.append({"image": name, "level": grade})

    labels_csv = os.path.join(folder, "trainLabels.csv")
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    return image_dir, labels_csv


def make_synthetic_messidor(folder, num_images=10, seed=1):
    """Fake Messidor-2 images + csv, last row ungradable. Returns (image_dir, labels_csv)."""
    rng = np.random.RandomState(seed)

    image_dir = os.path.join(folder, "images")
    os.makedirs(image_dir, exist_ok=True)

    rows = []
    for i in range(num_images):
        grade = i % 5
        #7 -> "007"
        image_id = f"synthetic_{i:03d}.png"
        _draw_fake_fundus(grade, rng).save(os.path.join(image_dir, image_id))
        rows.append({"image_id": image_id, "adjudicated_dr_grade": grade,
                     "adjudicated_gradable": 1})

    #ungradable one, empty grade
    rows.append({"image_id": "synthetic_ungradable.png",
                 "adjudicated_dr_grade": None, "adjudicated_gradable": 0})

    labels_csv = os.path.join(folder, "messidor_data.csv")
    pd.DataFrame(rows).to_csv(labels_csv, index=False)

    return image_dir, labels_csv
