from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    transform_breast_tissue,
    transform_cleveland,
    transform_dermatology,
    transform_ecoli,
    transform_eucalyptus,
    transform_glass,
    transform_haberman,
    transform_heart,
    transform_iris,
    transform_led,
    transform_libras,
    transform_liver,
    transform_madelon,
    transform_new_thyroid_1,
    transform_new_thyroid_2,
    transform_page_blocks_1_3,
    transform_pima,
    transform_vehicle,
    transform_vowel,
    transform_wine,
    transform_yeast_1,
)


def create_pipeline(**kwargs) -> Pipeline:
    transform_funcs = [
        transform_breast_tissue,
        transform_cleveland,
        transform_dermatology,
        transform_ecoli,
        transform_eucalyptus,
        transform_glass,
        transform_haberman,
        transform_heart,
        transform_iris,
        transform_led,
        transform_libras,
        transform_liver,
        transform_madelon,
        transform_new_thyroid_1,
        transform_new_thyroid_2,
        transform_page_blocks_1_3,
        transform_pima,
        transform_vehicle,
        transform_vowel,
        transform_wine,
        transform_yeast_1,
    ]
    return pipeline(
        [
            node(
                func=func,
                inputs=f'{func.__name__.replace("transform_", "")}_data',
                outputs=f'{func.__name__.replace("transform_", "")}_numerical_features_binary_target_imbalanced_data',
                name=f'{func.__name__}_numerical_features_binary_target_imbalanced_data_node',
            )
            for func in transform_funcs
        ]
    )
