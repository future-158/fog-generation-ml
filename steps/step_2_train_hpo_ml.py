from pathlib import Path
from typing import *

import catboost
import hydra
import joblib
import lightgbm
import numpy as np
import xgboost
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import calc_metrics
from step_1_pre_train_ml import load_data
from hydra.utils import get_original_cwd

# from tune_sklearn import TuneSearchCV
def custom_cross_val_predict(model, X, y, cv, sample_weight, cfg) -> float:
    """
    model, x, y, cross validator, fit_params가 주어지면
    cv로 x,y를 n등분하고 각각 model로 fitting한 후
    전체 성능을 계산함
    """
    pred_list = []
    obs_list = []

    # iterate over split
    fit_params = {}
    for tidx, vidx in cv.split(X, y):
        # model이 nested pipeline이므로 steps[-1][0]은 estimator임
        fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight[tidx]

        usecols = [x for x in X.columns if x not in cfg.drop_cols]
        X = X[usecols]
        assert X.dtypes.value_counts().size == 1

        y = y.astype(int)

        def train():
            model.fit(
                X.iloc[tidx],
                y.iloc[tidx],
                **fit_params,
            )

            pred = model.predict(X.iloc[vidx])
            assert pred.ndim == 1

            pred_list.append(pred)
            obs_list.append(y.iloc[vidx])

        # call train function defined above.
        train()

    # concatenate cross validation result (1d numpy array)
    pred = np.concatenate(pred_list)
    obs = np.concatenate(obs_list)

    # ignore na
    selection_mask = np.logical_not(np.isnan(obs))
    return calc_metrics(obs[selection_mask], pred[selection_mask])



def refit(model, X, y, sample_weight, cfg) -> float:
    usecols = [x for x in X.columns if x not in cfg.drop_cols]
    X = X[usecols]

    assert X.dtypes.value_counts().size == 1
    y = y.astype(int)
    
    fit_params = {}
    fit_params[f"{model.steps[-1][0]}__sample_weight"] = sample_weight

    model.fit(
        X,
        y,
        **fit_params,
    )
    return model




# get random forest
def get_rf(model_params: Dict):
    """
    lightgbm with randomforest mode
    """
    return lightgbm.LGBMClassifier(
        device="gpu",
        boosting="rf",
        **model_params,
    )


# get lightgbm
def get_lgb(model_params: Dict):
    """
    lightgbm with gbdt mode
    """
    return lightgbm.LGBMClassifier(
        device="gpu", boosting="gbdt", **model_params
    )


def get_cb(model_params: Dict):
    """
    gpu catboost
    """

    return catboost.CatBoostClassifier(
        learning_rate=0.1,
        # n_estimators=100,
        task_type="GPU",
        devices="0",
        gpu_ram_part=0.33,
        **model_params,
    )


# get xgboost
def get_xgb(model_params: Dict):
    return xgboost.XGBClassifier(
        tree_method="gpu_hist",
        use_label_encoder=False,
        **model_params,
    )


# get estimator
def get_estimator(model_name, model_params: Optional[Dict] = None):
    if model_params is None:
        model_params = {}

    if model_name == "rf":
        return get_rf(model_params)

    elif model_name == "cb":
        return get_cb(model_params)

    elif model_name == "xgb":
        return get_xgb(model_params)

    elif model_name == "lgb":
        return get_lgb(model_params)


# load scikit-learn pipeline
def load_pipe(cfg: DictConfig):
    base_model = get_estimator(cfg.model_name, cfg.model_params)
    pipe = make_pipeline(
        StandardScaler(),
        base_model,
    )
    return pipe

def _main(cfg: Union[DictConfig, OmegaConf]):
    if cfg.stage == 'train':
        X, y, cv = load_data(cfg)

        sample_weight = np.ones_like(y)
        for station_code, pos_label_weight in cfg.pos_label_weights.items():
            mask = np.logical_and(y == 1, X.station_code == station_code)
            sample_weight[mask] = pos_label_weight

        model = load_pipe(cfg)
        metrics = custom_cross_val_predict(model, X, y, cv, sample_weight,cfg)
        OmegaConf.save(
            OmegaConf.create(metrics),
            'metrics.yml'
        )
        return metrics['CSI']

    else:
        assert cfg.stage == 'refit'
        X, y = load_data(cfg)
        sample_weight = np.ones_like(y)
        for station_code, pos_label_weight in cfg.pos_label_weights.items():
            mask = np.logical_and(y == 1, X.station_code == station_code)
            sample_weight[mask] = pos_label_weight

        model = load_pipe(cfg)
        model = refit(model, X, y, sample_weight,cfg)

        dest = Path(get_original_cwd()) / cfg.catalogue.model
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            model,
            dest
            
        )
        return
        

@hydra.main(config_path="../conf", config_name="config.yml")
def main(cfg: DictConfig) -> None:
    return _main(cfg)
    
if __name__ == "__main__":
    main()

