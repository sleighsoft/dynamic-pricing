import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd


# Windows install:
# http://www.picnet.com.au/blogs/guido/post/2016/09/22/xgboost-windows-x64-binaries-for-download/


def get_model(conf):
  """Returns an XGBClassifier model with the given parameters set.

  args:
    conf: A dict of model parameter names and values
  """
  model = XGBClassifier(**conf)
  return model


def find_n_estimators(model_params, X, Y, conf):
  """Finds the best number of estimators (trees) for the training data and
  provided model parameters."""
  model = get_model(model_params)
  xgb_param = model.get_xgb_params()
  xgtrain = xgb.DMatrix(X, label=Y)
  # When cv stops we have found a good number of estimators
  cvresult = xgb.cv(
      xgb_param,
      xgtrain,
      num_boost_round=model.get_params()['n_estimators'],
      nfold=conf['nfold'],
      metrics='auc',
      early_stopping_rounds=conf['early_stopping_rounds'])
  return {'n_estimators': cvresult.shape[0]}
