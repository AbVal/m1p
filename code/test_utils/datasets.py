import numpy as np
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, LabelEncoder


def get_splits(X, y, seed=42):
    train_X, rest_X, train_y, rest_y = train_test_split(
      X,
      y,
      train_size=0.7,
      shuffle=True,
      random_state=seed
    )

    # 10% test and 20% validation data
    valid_X, test_X, valid_y, test_y = train_test_split(
      rest_X, rest_y,
      test_size=0.3333,
      shuffle=True,
      random_state=seed
    )

    return train_X, train_y, valid_X, valid_y, test_X, test_y


def handle_categorial(data, seed=42):

    train_X, train_y, valid_X, valid_y, test_X, test_y = data

    cat_cols = train_X.select_dtypes(['category', object]).columns

    if list(cat_cols):
        target_enc = TargetEncoder()
        train_X[cat_cols] = target_enc.fit_transform(train_X[cat_cols], train_y)

        valid_X[cat_cols] = target_enc.transform(valid_X[cat_cols])
        test_X[cat_cols] = target_enc.transform(test_X[cat_cols])

    return train_X, train_y, valid_X, valid_y, test_X, test_y


def prepare_cali_housing(seed=33):
    raw_data = fetch_california_housing(as_frame=True)
    X = raw_data.get("data")
    y = np.log(raw_data.get("target"))

    data = get_splits(X, y, seed=seed)
    data = handle_categorial(data, seed=seed)

    return dict(data=data, name='California Housing')


def prepare_openml(data_id, name, seed=42):
    X, y = fetch_openml(data_id=data_id, as_frame=True, return_X_y=True)

    if len(np.unique(y)) == 2:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
        print('Transformed target to:', np.unique(y))

    data = get_splits(X, y, seed=seed)
    data = handle_categorial(data, seed=seed)

    return dict(data=data, name=name)


def prepare_credit_g(seed=42):
    return prepare_openml(31, 'credit-g', seed)


def prepare_steel_plates(seed=42):
    return prepare_openml(1504, 'steel_plates', seed)


def prepare_climate_crashes(seed=42):
    return prepare_openml(1467, 'climate_crashes', seed)


def prepare_scene(seed=42):
    return prepare_openml(312, 'scene', seed)


def prepare_spam(seed=42):
    return prepare_openml(44, 'spam', seed)
