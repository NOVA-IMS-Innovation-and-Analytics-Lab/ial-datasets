from collections import Counter

import pandas as pd
from imblearn.datasets import make_imbalance

FACTOR_MAPPING = {
    'breast_tissue': [1, 2, 3, 4],
    'cleveland': [1],
    'dermatology': [1, 2],
    'ecoli': [1, 2, 3, 4, 5],
    'eucalyptus': [1, 2, 3, 4, 5],
    'glass': [1, 2, 3, 4, 5],
    'haberman': [1, 2, 3, 4, 5],
    'heart': [1, 2, 3, 4, 5],
    'iris': [1, 2, 3, 4, 5],
    'led': [1, 2, 3, 4],
    'libras': [1, 2, 3],
    'liver': [1, 2, 3, 4, 5],
    'madelon': [1, 2, 3, 4, 5],
    'new_thyroid_1': [1, 2, 3, 4],
    'new_thyroid_2': [1, 2, 3, 4],
    'page_blocks_1_3': [1, 2, 3],
    'pima': [1, 2, 3, 4, 5],
    'vehicle': [1, 2, 3, 4, 5],
    'vowel': [1, 2, 3, 4, 5],
    'wine': [1, 2, 3, 4, 5],
    'yeast_1': [1, 2, 3, 4, 5],
}


def make_data_imbalanced(data, params, factor):
    ratio = Counter(data['target']).copy()
    ratio[1] = int(ratio[1] / factor)
    X_imb, y_imb = make_imbalance(
        data.drop(columns='target'),
        data['target'],
        sampling_strategy=ratio,
        random_state=params['random_state'],
    )
    data_imbalanced = pd.concat([X_imb, y_imb], axis=1).reset_index(drop=True)
    return data_imbalanced


def generate_process_funcs():
    process_funcs = []
    for data_name, factors in FACTOR_MAPPING.items():
        for factor in factors:
            process_funcs.append(
                (
                    data_name,
                    factor,
                    lambda data, params, factor=factor: make_data_imbalanced(
                        data, params, factor
                    ),
                )
            )
    return process_funcs
