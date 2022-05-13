from pathlib import Path
from typing import *

from hydra.utils import get_original_cwd
import hydra
import joblib
import numpy as np
from omegaconf import DictConfig, OmegaConf

# from tune_sklearn import TuneSearchCV
from utils import calc_metrics
from step_2_train_hpo_ml import _main
from step_1_pre_train_ml import load_data
# evaluation function
def evaluation_fn(model, X, y):
    test_data = joblib.load(cfg.data.interim.test_path)
    X = test_data['X']
    y = test_data['y']
    pred = model.predict(X)
    # ignore na
    selection_mask = np.logical_not(np.isnan(y))
    model_path = Path(cfg.root) / "model" / cfg.station
    joblib.dump(model, model_path)
    # calc metrics
    metrics = calc_metrics(y[selection_mask], pred[selection_mask])
    payload = {}
    payload["pred"] = pred
    payload["test"] = y
    payload["metrics"] = metrics
    # return payload
    return payload

@hydra.main(config_path="../conf", config_name="config.yml")
def main(cfg: DictConfig) -> None:
    # cfg = OmegaConf.load('conf/ml.yaml')
    # define file pattern

    
    config_files = [
        config_file for config_file in 
        (
        Path(get_original_cwd()) / cfg.log_prefix
        ).glob('**/.hydra/config.yaml')
        if OmegaConf.load(config_file).api_version == cfg.api_version
        and OmegaConf.load(config_file).stage == 'train'
    ]

    metric_files = [
        config_file.parent.parent / 'metrics.yml' for config_file in config_files
    ]


    metric_files = [x for x in metric_files if x.exists()]
    best_metric_file = sorted(
        metric_files,
        key = lambda metric_file: OmegaConf.load(metric_file).CSI 
    )[-1]

    best_config_file = best_metric_file.parent / '.hydra' / 'config.yaml'
    cfg = OmegaConf.load(best_config_file)
    cfg.stage = 'refit'
    _main(cfg=cfg)

    cfg.stage = 'test'

    model = joblib.load(
        Path(get_original_cwd()) / cfg.catalogue.model
    )
    X_test, y_test = load_data(cfg)


    
    metrics = calc_metrics(y_test,pred = model.predict(X_test))
    metrics['stage'] = 'test'
    dest = Path(get_original_cwd()) / cfg.catalogue.test_result
    dest.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        OmegaConf.create(metrics),
        dest
    )

if __name__ == "__main__":
    main()
