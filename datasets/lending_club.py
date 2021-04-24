import os
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from urllib.error import URLError
from typing import Optional, Callable, Dict, Any

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.datasets.utils import (check_integrity,
                                        download_and_extract_archive)


@lru_cache()
def load_accepted(path):
    """Load 36 month loans."""
    df = pd.read_csv(path, usecols=['issue_d',
                                    'fico_range_low',
                                    'fico_range_high',
                                    'dti',
                                    'loan_amnt',
                                    'emp_length',
                                    'addr_state',
                                    'term',
                                    'loan_status'])

    df['date'] = pd.to_datetime(df['issue_d'], format='%b-%Y')
    df['fico'] = 0.5 * (df['fico_range_low'] + df['fico_range_high'])

    # Filter based on loan term
    df = df.loc[df.term == ' 36 months', :]
    return df

@lru_cache()
def load_rejected(path):
    """"Load rejected applications"""

    def p2f(x):
        return float(x.strip('%')) / 100

    df = pd.read_csv(path, usecols=['Debt-To-Income Ratio',
                                    'Amount Requested',
                                    'Risk_Score',
                                    'Employment Length',
                                    'State',
                                    'Application Date'],
                     converters={'Debt-To-Income Ratio': p2f})

    df = df.rename(columns={'Debt-To-Income Ratio': 'dti',
                            'Amount Requested': 'loan_amnt',
                            'Risk_Score': 'fico',
                            'Employment Length': 'emp_length',
                            'State': 'addr_state',
                            'Application Date': 'date'})
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    return df


class LendingClub(Dataset):
    """Lending Club's loans and/or rejected applications for the period between
    January 2009 through September 2012. Only 36-month loans are considered.

    The data was taken from Kaggle web site:
        https://www.kaggle.com/wordsforthewise/lending-club

    The number of accepted applications and number of defaults match exactly
    as in https://arxiv.org/pdf/1904.11376.pdf.

    The number of rejected applications does not match exactly.

    Args:
        root: Data directory.
        accepted: If True, dataset is loans, otherwise rejected applications.
        partition: "train", "dev", "test" or None.
        transform: Callable data transformer.
        download: Download data.
    """

    mirrors = [
        'https://datasets-74cc9685.s3.amazonaws.com/'
    ]

    resources = [
        ("lending_club.accepted_2007_to_2018Q4.csv.gz", 'e2d05a00a4b854a58e67fdaacebb2b63'),
        ("lending_club.rejected_2007_to_2018Q4.csv.gz", '008280a859da83b2fa5ad68cc85f68f7'),
    ]

    numeric_features = ['dti', 'loan_amnt', 'fico']
    categorical_features = ['emp_length', 'addr_state']

    # Keep loans (rejected applications) created in this time window
    date_start = datetime(2009, 1, 1)
    date_end = datetime(2012, 10, 1)

    # most recent quarter
    test_set_quarter = '2012Q3'

    # second most recent quarter
    dev_set_quarter = '2012Q2'

    def __init__(
        self,
        root: str,
        accepted: bool=True,
        partition: Optional[str]=None,
        transform: Optional[Callable]=None,
        download: bool = False
    ) -> None:
        self.root = root
        self.partition = partition
        self.accepted = accepted
        self.transform = transform
        assert partition in {'train', 'dev', 'test', None}

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.X_numeric, self.X_cat, self.y = self._load_data()
        assert len(self.X_numeric) == len(self.X_cat) == len(self.y)

    def __getitem__(self, index: Any) -> Dict[str, Any]:
        """
        Args:
            index (int): Index
        Returns:
            dict with feature arrays and targets
        """
        sample = {'X_numeric': self.X_numeric[index, :],
                  'X_categorical': self.X_cat[index, :],
                  'target': self.y[index]}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.y)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(
                self.raw_folder, os.path.splitext(os.path.basename(url))[0])
            ) for url, _ in self.resources
        )

    def _get_path(self):
        i = 0 if self.accepted else 1
        file_name = Path(self.resources[i][0]).stem
        return os.path.join(self.raw_folder, file_name)

    def _load_data(self):
        path = self._get_path()
        df = load_accepted(path) if self.accepted else load_rejected(path)
        df = df.loc[df.date >= self.date_start, :]
        df = df.loc[df.date < self.date_end, :]

        # Quarter of loan issuance or loan application
        df['quarter'] = pd.PeriodIndex(pd.to_datetime(df['date']),
                                       freq='Q').astype(str)

        # Define target variable
        df['target'] = np.nan

        if self.accepted:
            df.loc[df.loan_status == 'Charged Off', 'target'] = 1
            df.loc[df.loan_status == 'Fully Paid', 'target'] = 0
            df = df.dropna()
            df['target'] = df['target'].astype(int)

        if self.partition == 'train':
            df = df.loc[~df.quarter.isin([self.test_set_quarter,
                                          self.dev_set_quarter]), :]
        if self.partition == 'dev':
            df = df.loc[df.quarter == self.dev_set_quarter, :]

        if self.partition == 'test':
            df = df.loc[df.quarter == self.test_set_quarter, :]

        return (df[self.numeric_features].values,
                df[self.categorical_features].values,
                df['target'].values)

    def download(self) -> None:
        """Download data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.raw_folder,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))
