from io import BytesIO, StringIO
from re import sub
from string import ascii_lowercase
from urllib.parse import urljoin
from zipfile import ZipFile

import pandas as pd
import requests


def download_abalone(params):
    """Download the Abalone Data Set.

    https://archive.ics.uci.edu/ml/datasets/Abalone
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['abalone']
    )
    data = pd.read_csv(url, header=None)
    return data


def download_acute(params):
    """Download the Acute Inflammations Data Set.

    https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['acute']
    )
    data = pd.read_csv(url, header=None, sep='\t', decimal=',', encoding='UTF-16')
    return data


def download_adult(params):
    """Download the Adult Data Set.

    https://archive.ics.uci.edu/ml/datasets/Adult
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['adult']
    )
    data = pd.read_csv(url, header=None, na_values=' ?')
    return data


def download_annealing(params):
    """Download the Annealing Data Set.

    https://archive.ics.uci.edu/ml/datasets/Annealing
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['annealing']
    )
    data = pd.read_csv(url, header=None, na_values='?')
    return data


def download_arcene(params):
    """Download the Arcene Data Set.

    https://archive.ics.uci.edu/ml/datasets/Arcene
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['arcene'],
    )
    data, labels = [], []
    for data_type in ('train', 'valid'):
        data.append(
            pd.read_csv(
                urljoin(url, f'ARCENE/arcene_{data_type}.data'),
                header=None,
                sep=' ',
            )
        )
        labels.append(
            pd.read_csv(
                urljoin(
                    url,
                    ('ARCENE/' if data_type == 'train' else '')
                    + f'arcene_{data_type}.labels',
                ),
                header=None,
            )
        )
    data = pd.concat(data, ignore_index=True)
    labels = pd.concat(labels, ignore_index=True).rename(columns={0: data.shape[1] + 1})
    data = pd.concat([data, labels], axis=1)
    return data


def download_audit(params):
    """Download the Audit Data Set.

    https://archive.ics.uci.edu/ml/datasets/Audit+Data
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['audit'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('audit_data/audit_risk.csv').decode('utf-8')
    )
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), engine='python')
    return data


def download_banknote_authentication(params):
    """Download the Banknote Authentication Data Set.

    https://archive.ics.uci.edu/ml/datasets/banknote+authentication
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls'][
            'banknote_authentication'
        ],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_breast_cancer(params):
    """Download the Breast Cancer Wisconsin Data Set.

    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['breast_cancer'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_breast_tissue(params):
    """Download the Breast Tissue Data Set.

    http://archive.ics.uci.edu/ml/datasets/breast+tissue
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls'][
            'breast_tissue'
        ],
    )
    data = pd.read_excel(url, sheet_name='Data')
    return data


def download_contraceptive(params):
    """Download the Contraceptive Method Choice Data Set.

    https://archive.ics.uci.edu/ml/datasets/Contraceptive+Method+Choice
    """
    url = urljoin(
        params['uci_url'],
        params['mixed_features_binary_target_data_urls']['contraceptive'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_credit_approval(params):
    """Download the Credit Approval Data Set.

    https://archive.ics.uci.edu/ml/datasets/Credit+Approval
    """
    url = urljoin(
        params['uci_url'],
        params['mixed_features_binary_target_data_urls']['credit_approval'],
    )
    data = pd.read_csv(url, header=None, na_values='?')
    return data


def download_echocardiogram(params):
    """Download the Echocardiogram Data Set.

    https://archive.ics.uci.edu/ml/datasets/Echocardiogram
    """
    url = urljoin(
        params['uci_url'],
        params['mixed_features_binary_target_data_urls']['echocardiogram'],
    )
    data = pd.read_csv(
        url,
        header=None,
        error_bad_lines=False,
        warn_bad_lines=False,
        na_values='?',
    )
    return data


def download_flags(params):
    """Download the Flags Data Set.

    https://archive.ics.uci.edu/ml/datasets/Flags
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['flags']
    )
    data = pd.read_csv(url, header=None)
    return data


def download_heart_disease(params):
    """Download the Heart Disease Data Set.

    https://archive.ics.uci.edu/ml/datasets/Heart+Disease
    """
    urls = urljoin(
        params['uci_url'],
        params['mixed_features_binary_target_data_urls']['heart_disease'],
    )
    data = pd.concat(
        [pd.read_csv(url, header=None, na_values='?') for url in urls],
        ignore_index=True,
    )
    return data


def download_hepatitis(params):
    """Download the Hepatitis Data Set.

    https://archive.ics.uci.edu/ml/datasets/Hepatitis
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['hepatitis']
    )
    data = pd.read_csv(url, header=None, na_values='?')
    return data


def download_german_credit(params):
    """Download the German Credit Data Set.

    https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
    """
    url = urljoin(
        params['uci_url'],
        params['mixed_features_binary_target_data_urls']['german_credit'],
    )
    data = pd.read_csv(url, header=None, sep=' ')
    return data


def download_cleveland(params):
    """Download the Heart Disease Cleveland Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=980
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls']['cleveland'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('cleveland-0_vs_4.dat').decode('utf-8')
    )
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_dermatology(params):
    """Download the Dermatology Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=1330
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls']['dermatology'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('dermatology-6.dat').decode('utf-8')
    )
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_ecoli(params):
    """Download the Ecoli Data Set.

    https://archive.ics.uci.edu/ml/datasets/ecoli
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['ecoli'],
    )
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    return data


def download_eucalyptus(params):
    """Download the Eucalyptus Data Set.

    https://www.openml.org/d/188
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['eucalyptus'],
    )
    data = pd.read_csv(url)
    return data


def download_glass(params):
    """Download the Glass Identification Data Set.

    https://archive.ics.uci.edu/ml/datasets/glass+identification
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['glass'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_haberman(params):
    """Download the Haberman's Survival Data Set.

    https://archive.ics.uci.edu/ml/datasets/Haberman's+Survival
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['haberman'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_heart(params):
    """Download the Heart Data Set.

    http://archive.ics.uci.edu/ml/datasets/statlog+(heart)
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['heart'],
    )
    data = pd.read_csv(url, header=None, delim_whitespace=True)
    return data


def download_ionosphere(params):
    """Download the Ionosphere Data Set.

    https://archive.ics.uci.edu/ml/datasets/ionosphere
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['ionosphere'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_iris(params):
    """Download the Iris Data Set.

    https://archive.ics.uci.edu/ml/datasets/iris
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['iris'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_led(params):
    """Download the LED Display Domain Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=998
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls']['led'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data))
        .read('led7digit-0-2-4-5-6-7-8-9_vs_1.dat')
        .decode('utf-8')
    )
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_libras(params):
    """Download the Libras Movement Data Set.

    https://archive.ics.uci.edu/ml/datasets/Libras+Movement
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['libras'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_liver(params):
    """Download the Liver Disorders Data Set.

    https://archive.ics.uci.edu/ml/datasets/liver+disorders
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['liver'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_madelon(params):
    """Download the Arcene Data Set.

    https://archive.ics.uci.edu/ml/datasets/Madelon
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['madelon'],
    )
    X = pd.read_csv(url.format('data'), header=None, sep=' ')
    y = pd.read_csv(url.format('labels'), header=None, sep=' ')
    data = pd.concat([X, y], axis=1)
    return data


def download_new_thyroid_1(params):
    """Download the Thyroid 1 Disease Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=145
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls'][
            'new_thyroid_1'
        ],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('new-thyroid1.dat').decode('utf-8')
    )
    data = pd.read_csv(
        StringIO(sub(r'@.+\n+', '', unzipped_data)),
        header=None,
        sep=', ',
        engine='python',
    )
    return data


def download_new_thyroid_2(params):
    """Download the Thyroid 2 Disease Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=146
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls'][
            'new_thyroid_2'
        ],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('newthyroid2.dat').decode('utf-8')
    )
    data = pd.read_csv(
        StringIO(sub(r'@.+\n+', '', unzipped_data)),
        header=None,
        sep=', ',
        engine='python',
    )
    return data


def download_page_blocks_1_3(params):
    """Download the Page Blocks 1-3 Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=124
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls'][
            'page_blocks_1_3'
        ],
    )
    zipped_data = requests.get(url).content
    unzipped_data = (
        ZipFile(BytesIO(zipped_data)).read('page-blocks-1-3_vs_4.dat').decode('utf-8')
    )
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_parkinsons(params):
    """Download the Parkinsons Data Set.

    https://archive.ics.uci.edu/ml/datasets/parkinsons
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['parkinsons'],
    )
    data = pd.read_csv(url)
    return data


def download_pima(params):
    """Download the Pima Indians Diabetes Data Set.

    https://www.kaggle.com/uciml/pima-indians-diabetes-database
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['pima'],
    )
    data = pd.read_csv(url, header=None, skiprows=9)
    return data


def download_spambase(params):
    """Download and transform the Spambase Data Set.

    https://archive.ics.uci.edu/ml/datasets/Spambase
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_balanced_data_urls']['spambase'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_vehicle(params):
    """Download the Vehicle Silhouettes Data Set.

    https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes)
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['vehicle'],
    )
    data = []
    for letter in ascii_lowercase[0:9]:
        partial_data = pd.read_csv(
            urljoin(url, 'xa%s.dat' % letter),
            header=None,
            delim_whitespace=True,
        )
        partial_data = partial_data.rename(columns={18: 'target'})
        partial_data['target'] = partial_data['target'].isin(['van']).astype(int)
        data.append(partial_data)
    data = pd.concat(data)
    return data


def download_vowel(params):
    """Download the Vowel Recognition Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=127
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls']['vowel'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('vowel0.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_wine(params):
    """Download the Wine Data Set.

    https://archive.ics.uci.edu/ml/datasets/wine
    """
    url = urljoin(
        params['uci_url'],
        params['numerical_features_binary_target_imbalanced_data_urls']['wine'],
    )
    data = pd.read_csv(url, header=None)
    return data


def download_yeast_1(params):
    """Download the Yeast 1 Data Set.

    http://sci2s.ugr.es/keel/dataset.php?cod=153
    """
    url = urljoin(
        params['numerical_features_binary_target_imbalanced_data_urls']['keel'],
        params['numerical_features_binary_target_imbalanced_data_urls']['yeast_1'],
    )
    zipped_data = requests.get(url).content
    unzipped_data = ZipFile(BytesIO(zipped_data)).read('yeast1.dat').decode('utf-8')
    data = pd.read_csv(StringIO(sub(r'@.+\n+', '', unzipped_data)), header=None)
    return data


def download_thyroid(params):
    """Download the Thyroid Disease Data Set.

    https://archive.ics.uci.edu/ml/datasets/Thyroid+Disease
    """
    url = urljoin(
        params['uci_url'], params['mixed_features_binary_target_data_urls']['thyroid']
    )
    data = pd.read_csv(url, header=None, na_values='?')
    return data
