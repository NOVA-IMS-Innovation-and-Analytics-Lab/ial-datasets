from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    transform_arcene,
    transform_audit,
    transform_banknote_authentication,
    transform_breast_cancer,
    transform_ionosphere,
    transform_parkinsons,
    transform_spambase,
)


def create_pipeline(**kwargs) -> Pipeline:
    transform_funcs = [
        transform_arcene,
        transform_audit,
        transform_banknote_authentication,
        transform_breast_cancer,
        transform_ionosphere,
        transform_parkinsons,
        transform_spambase,
    ]
    return pipeline(
        [
            node(
                func=func,
                inputs=f'{func.__name__.replace("transform_", "")}_data',
                outputs=f'{func.__name__.replace("transform_", "")}_numerical_features_binary_target_balanced_data',
                name=f'{func.__name__}_numerical_features_binary_target_balanced_data_node',
            )
            for func in transform_funcs
        ]
    )
