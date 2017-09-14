from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
import models.xgboost_impl as xgboost
import models.gboost_impl as gboost
import os
import pickle
import re
import filelock


def grid_search(model_module, model_params, X, Y, conf):
  params = conf['param_distributions']
  if isinstance(conf['param_distributions'], dict):
    params = [params]

  for p in params:
    search = GridSearchCV(
        model_module.get_model(model_params),
        param_grid=p,
        scoring=conf['scoring'],
        n_jobs=conf['n_jobs'],
        cv=conf['cv'],
        iid=False)
    search.fit(X, Y)
    model_params.update(search.best_params_)
  return model_module.get_model(model_params)


def random_search(model_module, model_params, X, Y, conf):
  params = conf['param_distributions']
  if isinstance(conf['param_distributions'], dict):
    params = [params]

  for p in params:
    search = RandomizedSearchCV(
        model_module.get_model(model_params),
        param_distributions=p,
        n_iter=conf['n_iter'],
        scoring=conf['scoring'],
        n_jobs=conf['n_jobs'],
        cv=conf['cv'],
        iid=False)
    search.fit(X, Y)
    model_params.update(search.best_params_)
  return model_module.get_model(model_params)


def perform_parameter_search(
        model_module,
        model_params,
        search_conf,
        X,
        Y):
  search_type = search_conf['search_type']
  search_params = search_conf[search_type]
  if search_type == 'random_search':
    model = random_search(
        model_module,
        model_params,
        X,
        Y,
        search_params)
  elif search_type == 'grid_search':
    model = grid_search(
        model_module,
        model_params,
        X,
        Y,
        search_params)
  return model


def get_model_module_by_name(model_name):
  module = None
  if model_name == 'xgb':
    module = xgboost
  elif model_name == 'gbc':
    module = gboost
  return module


def find_latest_checkpoint_id(model_dir, basename):
  last_checkpoint = -1
  for f in os.listdir(model_dir):
    if f.startswith(str(basename)):
      ext = os.path.splitext(f)[-1][1:]
      m = re.match('ckpt-(\d+)', ext)
      if m and int(m.group(1)) > last_checkpoint:
        last_checkpoint = int(m.group(1))
  return last_checkpoint if last_checkpoint > -1 else None


def _create_basename(model_type, pid):
  return '{model_type}-{pid}'.format(model_type=model_type, pid=pid)


def _create_checkpoint_name(basename, chkpt_id):
  return '{}.ckpt-{}'.format(basename, chkpt_id)


def export(directory, data, model_type, pid, message=''):
  """Exports data to a checkpoint file at directory named
  <model_type>-<pid>.ckpt-<id> where id is incremental based on already
  existing checkpoints in directory.
  """
  basename = _create_basename(model_type, pid)
  last_checkpoint_id = find_latest_checkpoint_id(directory, basename)
  if last_checkpoint_id is None:
    last_checkpoint_id = -1
  checkpoint_name = _create_checkpoint_name(basename, last_checkpoint_id + 1)
  path = os.path.join(directory, checkpoint_name)
  with filelock.FileLock(path + '-lock'):
    with open(path, 'wb') as f:
      print('Exporting {} for #{} to {}'.format(message, basename, f.name))
      pickle.dump(data, f)


def clear_dir(directory, ignore=[]):
  for f in os.listdir(directory):
    if f not in ignore:
      os.remove(os.path.join(directory, f))


def load_latest_checkpoint(directory, model_type, pid, message=''):
  basename = _create_basename(model_type, pid)
  last_checkpoint_id = find_latest_checkpoint_id(directory, basename)
  checkpoint = None
  if last_checkpoint_id is None:
    print('No checkpoint for {} in {}'.format(basename, directory))
  else:
    checkpoint_name = _create_checkpoint_name(basename, last_checkpoint_id)
    path = os.path.join(directory, checkpoint_name)
    with filelock.FileLock(path + '-lock'):
      with open(path, 'rb') as f:
        print('Loading {} for #{} from {}'.format(message, basename, f.name))
        checkpoint = pickle.load(f)
  return checkpoint, last_checkpoint_id


def find_all_pids_for_modeltype(directory, model_type):
  pids = []
  for f in os.listdir(directory):
    if f.startswith(model_type):
      m = re.match('.*-(\d+)\..*', f)
      if m:
        pids.append(m.group(1))
  return pids


def load_checkpoint(directory, model_type, pid, checkpoint_id, message=''):
  """Load a specific checkpoint"""
  basename = _create_basename(model_type, pid)
  checkpoint_name = _create_checkpoint_name(basename, checkpoint_id)
  path = os.path.join(directory, checkpoint_name)
  checkpoint = None
  if os.path.exists(path):
    with filelock.FileLock(path + '-lock'):
      with open(path, 'rb') as f:
        print('Loading {} for #{} from {}'.format(message, basename, f.name))
        checkpoint = pickle.load(f)
  return checkpoint


def load_all_latest_models_data(directory, model_type, pids=None):
  """Loads and returns the latest models data for a specific model type from
  disk. Can be used to load e.g. models, params, selectors

  Args:
    directory: The directory to search for models.
    model_type: The type of models to load e.g. xgb.
    pids: A list of product ids to load models for if available. If set to None
      it will automatically load models for each pid in the directory.
  Returns:
    A dictionary of models where the key is the pid.
  """
  models = {}
  if pids is None:
    pids = find_all_pids_for_modeltype(directory, model_type)
  for pid in pids:
    model, chkpt_id = load_latest_checkpoint(directory, model_type, pid,
                                             message=model_type)
    if model is not None:
      models.update({int(pid): (model, chkpt_id)})
  return models


def get_feature_selector(model, threshold):
  """Create a feature selector that can be applied to features using
  selector.transform(feature) to consistently choose features.

  Args:
    model: A pre-trained scikit model
    threshold: A threshold to be used to filter out features below it

  Returns:
    A sklearn.feature_selection.SelectFromModel instance.
  """
  # selector = SelectFromModel(model, threshold=threshold, prefit=True)
  from sklearn.model_selection import StratifiedKFold
  from sklearn.feature_selection import RFECV
  selector = RFECV(estimator=model, step=1, cv=StratifiedKFold(3),
                   scoring='neg_log_loss', verbose=1, n_jobs=4)
  return selector


# TODO Clear the oldest params and models checkpoints to save disk space (low prio)
