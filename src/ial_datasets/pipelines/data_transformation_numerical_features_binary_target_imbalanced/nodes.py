import numpy as np

from ...utils import transform_numeric_features_binary_target


def transform_breast_tissue(data):
    """Transform the Breast Tissue Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['Case #'], target_col='Class', target_vals=['car', 'fad']
    )


def transform_ecoli(data):
    """Transform the Ecoli Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['0'], target_col='8', target_vals=['pp']
    )


def transform_eucalyptus(data):
    """Transform the Eucalyptus Data Set."""
    drop_cols = [
        'Abbrev',
        'Rep',
        'Locality',
        'Map_Ref',
        'Latitude',
        'Altitude',
        'Frosts',
        'Sp',
        'PMCno',
    ]
    return transform_numeric_features_binary_target(
        data.replace('?', np.nan),
        drop_cols=drop_cols,
        target_col='Utility',
        target_vals=['best'],
    )


def transform_glass(data):
    """Transform the Glass Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['0'], target_col='10', target_vals=[1]
    )


def transform_haberman(data):
    """Transform the Haberman Data Set."""
    return transform_numeric_features_binary_target(
        data, target_col='3', target_vals=[2]
    )


def transform_heart(data):
    """Transform the Heart Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=[2])


def transform_iris(data):
    """Transform the Iris Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['Iris-setosa'])


def transform_madelon(data):
    """Transform the Madelon Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['500'], target_vals=[-1]
    )


def transform_libras(data):
    """Transform the Libras Movement Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=[1])


def transform_liver(data):
    """Transform the Liver Disorders Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=[1])


def transform_pima(data):
    """Transform the Pima Indians Diabetes Data Set."""
    return transform_numeric_features_binary_target(data)


def transform_vehicle(data):
    """Transform the Vehicle Silhouettes Data Set."""
    return transform_numeric_features_binary_target(data)


def transform_wine(data):
    """Transform the Wine Data Set."""
    return transform_numeric_features_binary_target(
        data, target_col='0', target_vals=[2]
    )


def transform_new_thyroid_1(data):
    """Transform the Thyroid 1 Disease Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_new_thyroid_2(data):
    """Transform the Thyroid 2 Disease Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_cleveland(data):
    """Transform the Heart Disease Cleveland Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_dermatology(data):
    """Transform the Dermatology Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_led(data):
    """Transform the LED Display Domain Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_page_blocks_1_3(data):
    """Transform the Page Blocks 1-3 Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=['positive'])


def transform_vowel(data):
    """Transform the Vowel Recognition Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=[' positive'])


def transform_yeast_1(data):
    """Transform the Yeast 1 Data Set."""
    return transform_numeric_features_binary_target(data, target_vals=[' positive'])
