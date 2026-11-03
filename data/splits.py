"""
Patient level, stratified, seeded split of EyePACS into train/val/test (80/10/10).

    Both eyes of a patient go in the same set, otherwise the model could
    recognise the patient instead of grading (data leakage).
    Each grade is split separately so rare grades 3/4 show up in val/test.

"""

import numpy as np


def split_by_patient(labels, val_fraction=0.1, test_fraction=0.1, seed=42):
    """Returns (train_df, val_df, test_df), no patient in more than one."""
    assert 0 <= val_fraction and 0 <= test_fraction, "fractions must be >= 0"
    assert val_fraction + test_fraction < 1, "val + test must leave training data"

    #one grade per patient = their worse eye
    patient_grade = labels.groupby("patient_id")["level"].max()

    #own seeded generator so the split is always the same
    rng = np.random.RandomState(seed)

    train_ids = []
    val_ids = []
    test_ids = []

    #split each grade on its own (stratified)
    for grade in sorted(patient_grade.unique()):
        #sorted first so only the seed decides the order
        ids = sorted(patient_grade[patient_grade == grade].index.tolist())

        rng.shuffle(ids)

        n_test = int(round(len(ids) * test_fraction))
        n_val = int(round(len(ids) * val_fraction))

        #first n_test -> test, next n_val -> val, rest -> train
        test_ids += ids[:n_test]
        val_ids += ids[n_test : n_test + n_val]
        train_ids += ids[n_test + n_val :]

    #isin picks both eyes of each patient
    train_df = labels[labels["patient_id"].isin(train_ids)].reset_index(drop=True)
    val_df = labels[labels["patient_id"].isin(val_ids)].reset_index(drop=True)
    test_df = labels[labels["patient_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df
