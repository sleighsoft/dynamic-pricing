from merchant_sdk.api import KafkaApi, PricewarsRequester
import config
import models.model_helper as model_helper
import models.stats as stats
import data.csvparser as csv
import data.feature as feature
from data.preprocess import sample_data

import argparse
import os
import time
import numpy as np
import sys

conf = config.get_config()


def parse_arguments():
  parser = argparse.ArgumentParser(
      description='Machine learning on PriceWars simulation data')
  parser.add_argument(
      '--buy',
      type=str,
      default=None,
      help='Path to buy_offers.csv for offline training')
  parser.add_argument(
      '--train',
      type=str,
      default=None,
      help='Path to market_situations.csv for offline training')
  parser.add_argument(
      '--merchant',
      type=str,
      default=None,
      help='The merchant_id used to identify own offers')
  parser.add_argument(
      '--merchant_token',
      type=str,
      default=None,
      help='The merchant token used to download data')
  parser.add_argument(
      '--test',
      type=str,
      default=None,
      help='Path to test_offers.csv used to test a trained model')
  parser.add_argument(
      '--output',
      type=str,
      default=None,
      help='Path to output predictions to')
  parser.add_argument(
      '--previous',
      action='store_true',
      default=False,
      help='If not set previously created situations file will be read')
  parser.add_argument(
      '--model_type',
      type=str,
      default='xgb',
      help='The model type to load when running with test')
  parser.add_argument(
      '--no_param_search',
      action='store_true',
      default=False,
      help='When set no parameter search e.g. grid search is performed')
  parser.add_argument(
      '--no_clear',
      action='store_true',
      default=False,
      help='When set the run/* directories will be cleared')
  parser.add_argument(
      '--pid',
      action='append',
      type=int,
      default=None,
      help='If --pid <a-pid> is passed in one or many times models for only'
      ' these pids will be trained. They have to be present in the training'
      ' data.')
  return parser.parse_args()


def make_relative_path(path):
  script_dir = os.path.dirname(os.path.realpath(__file__))
  return os.path.join(script_dir, path)


def load_and_create_training_data(
        market_situation_url,
        buy_offers_url,
        merchant_id):
  """Downloads marketSituation.csv and buyOffer.csv and computes features.
  """
  try:
    print('Downloading marketSituation.csv from {}'.format(market_situation_url))
    market_situations = csv.read_csv(market_situation_url)
    print('Done downloading marketSituation.csv')
    print('Downloading buyOffer.csv from {}'.format(buy_offers_url))
    buy_offers = csv.read_csv(buy_offers_url)
    print('Done downloading buyOffer.csv')
  except Exception as e:
    print('[Error] CSV download failed: {}'.format(e))
    sys.exit()
  print('Computing features')
  situations = feature.compute_situations_dict(
      market_situations,
      merchant_id,
      buy_offers)
  return situations


def train_and_export(situations, param_search, pids=None):
  """Trains a model for each product id found in situations and exports it.
  """
  labels = np.array([v['label'] for v in situations.values()])
  product_ids = [v['product_id'] for v in situations.values()]
  distinct_ids = set(product_ids)
  if pids is not None and distinct_ids.issuperset(pids):
    print('[Training] Training on subset of all pids: {}'.format(set(pids)))
    distinct_ids = set(pids)
  product_ids = np.array(product_ids)
  features = feature.get_list_of_features_from_situations_dict(situations)
  features = np.array(features)

  for pid in distinct_ids:
    print('[Training] Creating model for product #{}'.format(pid))
    indices = np.where(product_ids == pid)
    pid_features = features[indices]
    pid_labels = labels[indices]
    true_label = sum(pid_labels)
    print('[Training] Performing {} sampling with {}+{}={} samples'.format(
        conf['sampling_mode'], len(pid_labels) - true_label, true_label,
        len(pid_features)))
    pid_features_sampled, pid_labels_sampled = sample_data(
        pid_features, pid_labels, conf['sampling_mode'],
        conf['sampling_ratio'])
    print('[Training] Finished sampling with {} samples'.format(
        len(pid_features_sampled)))

    model_type = conf['model']
    model_conf = conf['model_config'][model_type]
    # The model_module should implement 'get_model(model_conf)'
    model_module = model_helper.get_model_module_by_name(model_type)

    if param_search:
      print('[Training] Performing parameter search')
      search_conf = conf['param_search'][model_type]

      start_time = time.time()
      if model_type == 'xgb' and search_conf['perform_estimator_search']:
        print('[Training] Searching for best number of estimators')
        n_estimators = model_module.find_n_estimators(
            model_conf,
            pid_features_sampled,
            pid_labels_sampled,
            search_conf['estimator_search'])
        print('[Training] Found n_estimators: {}'.format(n_estimators))
        model_conf.update(n_estimators)

      model = model_helper.perform_parameter_search(
          model_module,
          model_conf,
          search_conf,
          pid_features_sampled,
          pid_labels_sampled)
      print('[Trainig] Parameter search took {} seconds'.format(
          time.time() - start_time))
      print('[Training] Found parameters: {}'.format(model.get_params()))
      # Export model parameter
      model_helper.export(conf['param_save_dir'], model.get_params(),
                          model_type, pid,
                          message='params-{}'.format(model_type))
    else:
      # If no parameter search is performed load parameters from disk instead
      loaded_params, _ = model_helper.load_latest_checkpoint(
          conf['param_save_dir'], model_type, pid,
          message='params-{}'.format(model_type))
      if loaded_params is not None:
        print('[Training] Using model parameters from disk')
        model_conf = loaded_params
      model = model_module.get_model(model_conf)

    print("[Training] Fitting model {} on {} data points".format(
        model_type, len(pid_features_sampled)))
    start_time = time.time()
    model.fit(pid_features_sampled, pid_labels_sampled)
    model_timestamp = time.time()
    if conf['feature_selection']:
      print('[Training] Performing feature selection and re-training')
      selector = model_helper.get_feature_selector(
          model, conf['feature_selection_threshold'])
      selector.fit(pid_features_sampled, pid_labels_sampled)
      pid_features_sampled_selected = selector.transform(pid_features_sampled)
      print('[Training] Selecting {}/{} features'.format(
          pid_features_sampled_selected.shape[1],
          pid_features_sampled.shape[1]))
      pid_features = selector.transform(pid_features)
      model_helper.export(conf['selector_save_dir'],
                          (selector, model_timestamp), model_type,
                          pid, message='selector-{}'.format(model_type))
      feat_names = list([v['features'] for v in situations.values()][0].keys())
      print('[Training] Feature importances before selection: {}'.format(
          list(zip(feat_names, model.feature_importances_))))
      model.fit(pid_features_sampled_selected, pid_labels_sampled)
    print('[Trainig] Fitting model took {} seconds'.format(
        time.time() - start_time))
    if conf['log_statistics']:
      predictions = model.predict(pid_features)
      probabilities = model.predict_proba(pid_features)
      stats.print_all_scores(pid_labels, predictions, probabilities,
                             len(pid_features),
                             len(pid_labels), True)
      feat_names = list([v['features'] for v in situations.values()][0].keys())
      print('[Training] Final feature importances: {}'.format(
          list(zip(feat_names, model.feature_importances_))))

    # Export model
    model_helper.export(conf['model_save_dir'], (model, model_timestamp),
                        model_type, pid,
                        message='model-{}'.format(model_type))


def calculate_sales_probabilities(marketSituations, merchant_id, model_type):
  """Offline API to run sales probability calculation as requested on piazza.
  Expects n market situations and returns an array with n sales
  probabilities."""
  print('Calculating sales probabilities for test data')
  models_data = model_helper.load_all_latest_models_data(
      conf['model_save_dir'], model_type)
  models = {p: model for p, ((model, timestamp), ckpt_id)
            in models_data.items()}
  model_ckpt_ids = {p: ckpt_id for p, ((model, timestamp), ckpt_id)
                    in models_data.items()}
  selectors = {}
  for p, ckpt_id in model_ckpt_ids.items():
    selector = model_helper.load_checkpoint(
        conf['selector_save_dir'], model_type,
        p, ckpt_id, message='selector')
    if selector is not None:
      selectors.update({p: selector[0]})
    else:
      print('[Warning] Could not load selector for {}'.format(p))
  marketSituations = csv.read_csv(marketSituations)
  situations = feature.compute_situations_dict(marketSituations, merchant_id)
  features = feature.get_list_of_features_from_situations_dict(situations)
  features = np.array(features)
  product_ids = [v['product_id'] for v in situations.values()]
  distinct_ids = set(product_ids)
  product_ids = np.array(product_ids)
  offer_ids = np.array([v['offer_id'] for v in situations.values()])
  result = np.zeros([len(features), 2])
  for pid in distinct_ids:
    print('[Test] Predicting #{}'.format(pid))
    indices = np.where(product_ids == pid)
    pid_features = features[indices]
    model = models[pid]
    if conf['feature_selection']:
      selector = selectors[pid]
      pid_features_selected = selector.transform(pid_features)
      print('Selecting {}/{} features'.format(
          pid_features_selected.shape[1],
          pid_features.shape[1]))
      pid_features = pid_features_selected
    probabilities = model.predict_proba(pid_features)[:, 1]
    result[indices] = np.array((offer_ids[indices], probabilities)).T
  return result


if __name__ == '__main__':
  try:
    print('#### Start Market Learning ####')
    args = parse_arguments()

    merchant_id = args.merchant
    if merchant_id is None:
      merchant_id = config.calculate_merchant_id(args.merchant_token)

    if not args.no_clear:
      model_helper.clear_dir('run/data/', ignore=['.gitkeep'])
      model_helper.clear_dir('run/models/', ignore=['.gitkeep'])
      model_helper.clear_dir('run/params/', ignore=['.gitkeep'])

    offline = args.train is not None and args.buy is not None

    if offline:
      print('Using offline training files')
      market_situation_csv_url = args.train
      buy_offer_csv_url = args.buy
    else:
      PricewarsRequester.add_api_token(args.merchant_token)
      kafka_api = KafkaApi(
          host=conf['merchant_config']['kafka_reverse_proxy_url'])
      print('Using online training files')
      market_situation_csv_url = kafka_api.request_csv_export_for_topic(
          'marketSituation')
      buy_offer_csv_url = kafka_api.request_csv_export_for_topic('buyOffer')

    print('Loading and creating training data')
    situations = load_and_create_training_data(
        market_situation_csv_url,
        buy_offer_csv_url,
        merchant_id)

    if os.path.exists(conf['situations_save_path']) and args.previous:
      print('Loading previous features')
      old_situations = csv.load_situations_from_disk(
          conf['situations_save_path'])
      old_situations.update(situations)
      situations = old_situations
    else:
      csv.save_situations_to_disk(situations, conf['situations_save_path'])
    print('Training models')
    train_and_export(situations, not args.no_param_search, pids=args.pid)

    if (args.merchant is not None and args.test is not None):
      probs = calculate_sales_probabilities(args.test, args.merchant,
                                            args.model_type)
      output_str = str([(int(k), v) for k, v in list(map(tuple, probs))])
      if args.output is not None:
        if os.path.dirname(args.output) != '':
          os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w+') as f:
          f.write(output_str)
      else:
        print(output_str)
      model_helper.clear_dir('run/data/', ignore=['.gitkeep'])
      model_helper.clear_dir('run/models/', ignore=['.gitkeep'])
      model_helper.clear_dir('run/params/', ignore=['.gitkeep'])
      model_helper.clear_dir('run/selectors/', ignore=['.gitkeep'])
  except Exception as e:
    print('[Error] An exception occurred: {}'.format(e))
