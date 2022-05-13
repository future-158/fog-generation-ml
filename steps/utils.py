import argparse
import datetime
import enum
import functools
import io
import itertools
import os
import re
import sys
import time
import uuid
import warnings
from functools import partial
from inspect import signature
from io import BytesIO
from itertools import product
from os import listdir
from pathlib import Path
from random import sample
from subprocess import run
from types import SimpleNamespace
from typing import *

import joblib
import numpy as np
import omegaconf
import pandas as pd
from catboost.utils import eval_metric, get_gpu_device_count
from numpy.lib.stride_tricks import as_strided
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, jaccard_score, make_scorer,
                             mean_tweedie_deviance, precision_score,
                             recall_score, roc_auc_score)


def calc_metrics(obs: pd.Series, pred: np.ndarray, binary=True) -> Dict:
    """
    성능지표 계산. 2분류만 사용
    """

    if binary:  # 2분류
        tn, fp, fn, tp = confusion_matrix(obs, pred).flatten()
        metrics = dict(
            ACC=accuracy_score(obs, pred),
            CSI=jaccard_score(obs, pred),
            PAG=precision_score(obs, pred),
            POD=recall_score(obs, pred),
            F1=f1_score(obs, pred),
            TN=tn,
            FP=fp,
            FN=fn,
            TP=tp,
        )

    else:  # 3분류, macro micro 계산
        metrics = {}
        metrics["ACC"] = accuracy_score(obs, pred)
        metrics["macro_CSI"] = jaccard_score(obs, pred, average="macro")
        metrics["macro_PAG"] = precision_score(obs, pred, average="macro")
        metrics["macro_POD"] = recall_score(obs, pred, average="macro")
        metrics["macro_F1"] = f1_score(obs, pred, average="macro")

        metrics["micro_CSI"] = jaccard_score(obs, pred, average="micro")
        metrics["micro_PAG"] = precision_score(obs, pred, average="micro")
        metrics["micro_POD"] = recall_score(obs, pred, average="micro")
        metrics["micro_F1"] = f1_score(obs, pred, average="micro")

    metrics = { k: float(v) for k, v in metrics.items()}
    return metrics



cfg = OmegaConf.load('conf/station_code/SF_0003.yaml')


for file in list(Path('conf/station_code').glob('*.yaml')):
    station_code = file.stem
    if station_code == 'SF_0003':
        continue
    cfg = OmegaConf.load(file)
    cfg.station_code = station_code
    cfg.pos_label_weights = {}
    for node in cfg.neighbors:
        cfg.pos_label_weights[node] = 1

    cfg.hydra = {}
    cfg.hydra.sweeper = {}
    cfg.hydra.sweeper.search_space = {}

    for node in cfg.neighbors:
        cfg.hydra.sweeper.search_space[node] = {
            'type': 'int',
            'low': 0,
            'high': 50,
            'step': 1
        }
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.save(cfg, file)
    

