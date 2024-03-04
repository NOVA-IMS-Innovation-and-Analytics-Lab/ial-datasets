from ...utils import transform_numeric_features_binary_target


def transform_arcene(data):
    """Transform the Arcene Data Set."""
    return transform_numeric_features_binary_target(data, drop_cols=data.columns[1500:])


def transform_audit(data):
    """Transform the Audit Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['LOCATION_ID'], target_col='Risk'
    )


def transform_banknote_authentication(data):
    """Transform the Banknote Authentication Data Set."""
    return transform_numeric_features_binary_target(data)


def transform_breast_cancer(data):
    """Transform the Breast Cancer Wisconsin Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['0'], target_col='1', target_vals=['M']
    )


def transform_ionosphere(data):
    """Transform the Ionosphere Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['0', '1'], target_vals=['b']
    )


def transform_parkinsons(data):
    """Transform the Parkinsons Data Set."""
    return transform_numeric_features_binary_target(
        data, drop_cols=['name'], target_col='status', target_vals=[0]
    )


def transform_spambase(data):
    """Transform the Spambase Data Set."""
    return transform_numeric_features_binary_target(data)
