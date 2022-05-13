from pathlib import Path
from typing import *

import joblib
from hydra.utils import get_original_cwd
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
# from scipy.ndimage import (binary_closing, binary_dilation, binary_erosion,
#                            binary_opening)
from sklearn.model_selection import (PredefinedSplit,
                                     StratifiedGroupKFold
                                     )

class Custom_cv_neighbor:
    def __init__(
        self, station_code: str, groups: pd.Int64Index, n_splits: Optional[int] = 5
    ):
        self.station_code = station_code
        self.groups = groups  # east or west
        self.n_splits = n_splits
        self.gss = StratifiedGroupKFold(n_splits=n_splits)  # 비율맞춘 group split

    # return split size
    def get_n_splits(self):
        return self.n_splits

    def split(self, X, y, groups: Optional[list] = None, **kwargs):
        self_mask = X.station_code == self.station_code
        assert self_mask.sum() > 0  # sanity check
        for tidx, vidx in self.gss.split(X, y, self.groups):
            yield tidx, np.intersect1d(
                np.flatnonzero(self_mask), vidx
            )  # return index with mask


def load_data(
    cfg: Union[DictConfig, OmegaConf]
):
    # load data
    X_list = []
    y_list = []

    for i, station_code in enumerate([cfg.station_code, *cfg.neighbors]):
        source = Path(get_original_cwd()) / cfg.catalogue.processed
        source = source.with_name(f'{station_code}.pkl')

        X = joblib.load(source)["x"]    
        X = X.set_axis([f"{c1}_{c2}" for c1, c2 in X.columns], axis=1)
        y = joblib.load(source)["y"].reset_index()

        # set index
        y = y.set_index("datetime")
        X.index = y.index # duplicated index because it contains other stations.

    
        X = X.join(
            y[
                [
                    "time_day_sin",
                    "time_day_cos",
                    "time_year_sin",
                    "time_year_cos",
                    "std_ws",
                    "std_ASTD",
                    "std_rh",
                ]
            ]
        )


        y = y[cfg.target_name]
        

        # dropna
        drop_mask = X.isna().any(axis=1) | y.isna()
        if station_code != cfg.station_code:
            drop_mask |= (y != 1)

        X = X.iloc[np.flatnonzero(~drop_mask)]

        if cfg.transform.korea:
            y = transform_korea(y)
        y = y[~drop_mask]

        # sanity check
        assert not X.isna().sum().sum()    
        X['station_code'] = station_code
        X_list.append(X)
        y_list.append(y)


    X = pd.concat(X_list, ignore_index=False)
    y = pd.concat(y_list, ignore_index=False)
    
    test_start = cfg.test_start
    test_mask = np.logical_and(
        X.station_code == cfg.station_code, y.index >= test_start)
    
    # test는 항상 고정이므로 PredefinedSplit 사용
    pds = PredefinedSplit(np.where(test_mask, 0, -1))
    # iterate over split
    for i, (train_idx, test_idx) in enumerate(pds.split(X, y)):
        X, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train(train+val), test에서 Train set은 하루 단위를 unit으로 하여 stratified shuffle하여 사용함
        groups = y.index.year * 366 + y.index.dayofyear
        cv = Custom_cv_neighbor(cfg.station_code, groups)

        if cfg.stage == 'train':
            return X, y, cv
        elif cfg.stage == 'refit':
            return X, y
        else:
            assert cfg.stage == 'test'
            usecols = [x for x in X.columns if x not in cfg.drop_cols]
            return X_test[usecols], y_test
            
        


# 고려대식 라벨 스무딩 방법. 해무 사이의 비해무는 해무로 처리하고, 동떨어진 해무는 비해무로 처리
def transform_korea(vis: pd.Series) -> pd.Series:
    assert vis.index.to_series().diff().value_counts().n_unique() == 0
    # 해무이면서 최근 1시간 동안 해무 데이터가 2개 이상인 데이터 mask
    end_points = np.logical_and(vis.eq(1), vis.rolling(6).sum().ge(2))
    end_points = end_points[end_points].index.sort_values()

    # 해무이면서 향후 1시간 동안 해무 데이터가 2개 이상인 데이터 mask
    start_points = np.logical_and(
        vis.sort_index(ascending=False).eq(1),
        vis.sort_index(ascending=False).rolling(6).sum().ge(2),
    )

    start_points = start_points[start_points].index.sort_values()
    assert (end_points - start_points).max() <= pd.Timedelta(minutes=50)

    # 위 둘 사이를 해무로 채워 넣음
    for start_point, end_point in zip(start_points, end_points):
        assert start_point <= end_point
        vis[
            start_point:end_point
        ] = 1  # start_point, end_point 둘 다 어차피 1이라서 closed | open interval 상관없음

    drop_masks = np.logical_and(
        vis.sort_index(ascending=False).eq(1),
        vis.sort_index(ascending=False)
        .rolling(11, center=True)
        .sum()
        .eq(1),  # 0 0 0 0 0 1 0 0 0 0 0 경우 drop mask
    )

    drop_masks = drop_masks[drop_masks].index
    vis[drop_masks] = 0
    return vis

if __name__ == '__main__':
    essential_folderes = [
        'data/model',
        'data/processed',
        'data/hparams',
        'data/clean',
        'data/model_out',
        'data/log'
    ]
    for folder in essential_folderes:
        Path(folder).mkdir(parents=True, exist_ok=True)

    
