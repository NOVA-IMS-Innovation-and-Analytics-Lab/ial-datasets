from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    download_arcene,
    download_audit,
    download_banknote_authentication,
    download_breast_cancer,
    download_breast_tissue,
    download_cleveland,
    download_dermatology,
    download_ecoli,
    download_eucalyptus,
    download_glass,
    download_haberman,
    download_heart,
    download_ionosphere,
    download_iris,
    download_led,
    download_libras,
    download_liver,
    download_madelon,
    download_new_thyroid_1,
    download_new_thyroid_2,
    download_page_blocks_1_3,
    download_parkinsons,
    download_pima,
    download_spambase,
    download_vehicle,
    download_vowel,
    download_wine,
    download_yeast_1,
)


def create_pipeline(**kwargs) -> Pipeline:
    download_funcs = [
        download_arcene,
        download_audit,
        download_banknote_authentication,
        download_breast_cancer,
        download_breast_tissue,
        download_cleveland,
        download_dermatology,
        download_ecoli,
        download_eucalyptus,
        download_glass,
        download_haberman,
        download_heart,
        download_ionosphere,
        download_iris,
        download_led,
        download_libras,
        download_liver,
        download_madelon,
        download_new_thyroid_1,
        download_new_thyroid_2,
        download_page_blocks_1_3,
        download_parkinsons,
        download_pima,
        download_spambase,
        download_vehicle,
        download_vowel,
        download_wine,
        download_yeast_1,
    ]
    return pipeline(
        [
            node(
                func=func,
                inputs='parameters',
                outputs=f'{func.__name__.replace("download_", "")}_data',
                name=f'{func.__name__}_data_node',
            )
            for func in download_funcs
        ]
    )
