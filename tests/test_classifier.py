"""Tests for pathology feature extraction and the RandomForest classifier."""

import numpy as np
from sklearn.model_selection import train_test_split

from src.breathing_models import chest_displacement
from src.classifier import PathologyClassifier, build_dataset, extract_features


def test_feature_vector_shape_and_finite():
    _, d = chest_displacement("normal", duration=90, fs=100)
    feats = extract_features(d, 100)
    assert feats.shape == (6,)
    assert np.all(np.isfinite(feats))


def test_cheyne_stokes_has_higher_envelope_variation_than_normal():
    _, dn = chest_displacement("normal", duration=120, fs=100)
    _, dc = chest_displacement("cheyne_stokes", duration=120, fs=100)
    # feature index 2 is envelope coefficient of variation
    assert extract_features(dc, 100)[2] > extract_features(dn, 100)[2]


def test_classifier_accuracy_on_heldout_split():
    X, y = build_dataset(n_per_class=20, snr_db=(20.0, 10.0),
                         duration=90, fs=100, seed=0)
    x_tr, x_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=0)

    clf = PathologyClassifier(random_state=0).fit(x_tr, y_tr)

    assert clf.score(x_te, y_te) > 0.8
