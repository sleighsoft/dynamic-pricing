import argparse
import subprocess
import atexit
import numpy as np
import pandas as pd
import datetime
import config
import time
import random
import traceback
import queue
from models import model_helper
import data.feature as feature
from merchant_sdk import MerchantBaseLogic, MerchantServer
from merchant_sdk.api import PricewarsRequester, MarketplaceApi, ProducerApi
from merchant_sdk.models import Offer

conf = config.get_config()
subprocesses = []
subprocess_queue = queue.Queue()
param_search_pids = {}


def trigger_learning(merchant_token, pids_for_learning=None):
  """Trigger model training for specific pids or run next training in queue.
  This prevents running multiple training processes that consume all system
  resources.
  """
  # Remove finished subprocesses
  global subprocesses
  subprocesses = [p for p in subprocesses if p.poll() is None]
  if pids_for_learning is not None:
    subprocess_queue.put(pids_for_learning)
  if len(subprocesses) == 0 and subprocess_queue.qsize() > 0:
    pids_for_learning = subprocess_queue.get(False)
    print('{} learning events still in queue'.format(subprocess_queue.qsize()))
    print('Triggering model training @{} for {}'.format(
        datetime.datetime.now(), pids_for_learning))
    pid_command_list = []
    do_param_search = False
    for pid in pids_for_learning:
      if pid in param_search_pids:
        param_search_pids[pid] += 1
      else:
        param_search_pids[pid] = 1
      do_param_search = do_param_search or param_search_pids[pid] <= \
          conf['first_n_param_search']
      pid_command_list.append('--pid')
      pid_command_list.append(str(pid))
    command_list = (['python', 'demand_learning.py', '--no_clear',
                     '--previous', '--merchant_token', merchant_token] + pid_command_list)
    with open('learning.log', 'a+') as out:
      if do_param_search:
        proc = subprocess.Popen(command_list, stdout=out)
      else:
        command_list += ['--no_param_search']
        proc = subprocess.Popen(command_list, stdout=out)
      subprocesses.append(proc)


@atexit.register
def exit_subprocesses():
  for p in subprocesses:
    print('Terminating subprocess: {}'.format(p.pid))
    p.terminate()


class MLMerchant(MerchantBaseLogic):
  def __init__(self):
    MerchantBaseLogic.__init__(self)
    self.settings = conf['merchant_config']

    '''
      Predefined API token
    '''
    token = "{{API_TOKEN}}"
    if 'API_TOKEN' in token:
      token = conf['merchant_token']

    self.merchant_token = token
    self.merchant_id = config.calculate_merchant_id(self.merchant_token)

    '''
      Setup API
    '''
    PricewarsRequester.add_api_token(self.merchant_token)
    self.marketplace_api = MarketplaceApi(
        host=MerchantBaseLogic.get_marketplace_url())
    self.producer_api = ProducerApi(host=MerchantBaseLogic.get_producer_url())

    '''
      Setup Warmp-up phase
    '''
    self.warmup_count = {}
    pids = model_helper.find_all_pids_for_modeltype(
        conf['model_save_dir'], conf['model'])
    for pid in set(pids):
      pid = int(pid)
      print('[Debug] Found pre-trained model for {}'.format(pid))
      self.warmup_count.update({pid: conf['warmup']})
    self.warmup_time_tracker = {}

    '''
      Statistics
    '''
    self.sold_count = 0
    self.profit_count = 0
    self.uid_prices = {}
    self.previous_price_by_uid = {}

    '''
      Setup ML model
    '''
    self.last_learning = {}
    self.last_loading = datetime.datetime.now()

    '''
      Start Logic Loop
    '''
    self.run_logic_loop()

  def update_api_endpoints(self):
    """
    Updated settings may contain new endpoints, so they need to be set in
    the api client as well. However, changing the endpoint (after
    simulation start) may lead to an inconsistent state
    :return: None
    """
    self.marketplace_api.host = self.settings['marketplace_url']
    self.producer_api.host = self.settings['producer_url']

  '''
    Implement Abstract methods / Interface
  '''

  def update_settings(self, new_settings):
    MerchantBaseLogic.update_settings(self, new_settings)
    self.update_api_endpoints()
    return self.settings

  def sold_offer(self, offer):
    pid = offer.product_id
    print('[Debug] Sold product with id {}'.format(pid))
    self.sold_count += 1
    self.profit_count += offer.price - self.uid_prices[offer.uid]
    if self.sold_count % 40 == 0:
      print('[Statistics] Sold {} items for a total profit of {}'.format(
          self.sold_count, self.profit_count))
    if offer.uid in self.previous_price_by_uid:
      print('[Debug] Remove {} from price decline'.format(offer.uid))
      self.previous_price_by_uid.pop(offer.uid)
    if pid not in self.warmup_count:
      self.warmup_count[pid] = 1
      self.warmup_time_tracker[pid] = time.time()
    elif self.warmup_count[pid] < conf['warmup']:
      self.warmup_count[pid] += 1
      if self.warmup_count[pid] == conf['warmup']:
        print('[Debug] Warm-up for product {} took {} seconds'.format(
            pid, time.time() - self.warmup_time_tracker[pid]))

  '''
    Merchant Logic
  '''

  def price_product(
          self,
          model,
          own_product,
          product_prices_by_uid,
          current_offers=None,
          new=False,
          feature_selector=None):
    """
    Computes the price for a product. Maximizes the expected_profit by
    calculating a sell probability for prices between 1 and 80 with 0.5 steps.
    If model is set to None than a random price will be generated.
    """
    # Default to 10 in case of program restart which can mix up statistics
    purchase_price = self.uid_prices[own_product.uid] if \
        own_product.uid in self.uid_prices else 10.0
    # Set random prices for the first n products
    warmup = False
    pid = own_product.product_id
    if (pid not in self.warmup_count or
            self.warmup_count[pid] < conf['warmup']):
      warmup = True

    if warmup:
      print('[Debug] Warm-up {}/{} for product {}'.format(
          self.warmup_count[pid] if pid in self.warmup_count else 0,
          conf['warmup'],
          pid))

    if model is None or warmup:
      # Random price: There is no model for that type of product yet
      price = float(random.randrange(purchase_price * 10, 800) / 10)
      print('[Debug] Setting random price of {} to {}'.format(
          own_product.uid,
          price))
      return price

    offer_df = pd.DataFrame([o.to_dict() for o in current_offers])
    offer_df = offer_df[offer_df['product_id'] == pid]
    header = offer_df.keys()
    # Our product is the last element
    offer_df = offer_df.append(own_product.to_dict(), ignore_index=True)
    pandas_buffer = [i for i in offer_df.itertuples()]
    offer_dict = feature.pandas_buffer_to_product_dict(pandas_buffer, header)
    features = []
    for potential_price in np.arange(purchase_price + 0.5, 80.5, 0.5):
      # Extract features for each potential price
      offer_dict['price'][-1] = potential_price
      extracted_features = feature.extract_features_fn(
          offer_dict, -1)['features']
      features.append(extracted_features)
    data = np.array([list(v.values()) for v in features])

    if feature_selector is not None:
      data = feature_selector.transform(data)
    # Get sale probabilities for all possible prices
    sell_probs = model.predict_proba(data)[:, 1]
    # data[:, 0] = own_potential_prices
    profit_mean = np.mean(sell_probs)
    profit_mean_filter = sell_probs > profit_mean
    print('[Debug] Filtering sales probabilities < {}'.format(profit_mean))
    expected_profits = sell_probs * (data[:, 0] - purchase_price) * \
        profit_mean_filter
    price = data[expected_profits.argmax(), 0]
    if own_product.uid in self.previous_price_by_uid:
      if self.previous_price_by_uid[own_product.uid] <= price:
        print('[Debug] Decreasing price of {}'.format(own_product.uid))
        price = max(self.previous_price_by_uid[own_product.uid] -
                    conf['price_decrement'], self.uid_prices[own_product.uid])
        self.previous_price_by_uid.update({own_product.uid: price})
    else:
      self.previous_price_by_uid.update({own_product.uid: price})
    print('[Debug] Setting price of {} to {}'.format(
          own_product.uid,
          price))
    return price

  def _silently_update_offer(self, offer):
    try:
      self.marketplace_api.update_offer(offer)
      return True
    except Exception as e:
      print('[Error] Could not update an offer: {}'.format(
            traceback.format_exc()))
      return False

  def _silently_get_offers(self, include_empty_offers):
    try:
      # Get current market situation
      offers = self.marketplace_api.get_offers(include_empty_offers)
      return offers
    except Exception as e:
      print('[Error] Could not retrieve latest offers: {}'.format(
            traceback.format_exc()))
      return None

  def warmup_finished(self):
    if len(self.warmup_count) == 0:
      return []
    finished_pids = []
    for k, v in self.warmup_count.items():
      if v >= conf['warmup']:
        finished_pids.append(k)
    return finished_pids

  def maybe_run_demand_learning(self):
    finished_pids = self.warmup_finished()
    pids_for_learning = []
    for pid in finished_pids:
      if (pid not in self.last_learning or
              time.time() - self.last_learning[pid] >=
              self.settings['seconds_between_learning']):
        pids_for_learning.append(pid)
        self.last_learning[pid] = time.time()
    if len(pids_for_learning) > 0:
      trigger_learning(self.merchant_token, pids_for_learning)
    else:
      trigger_learning(self.merchant_token)

  def execute_logic(self):
    self.maybe_run_demand_learning()

    request_count = 0

    offers = self._silently_get_offers(True)
    if offers is None:
      # If offers cannot be retrieved skip everything and try again in 1 sec
      print('[Error] Retry in 1s')
      return 1.0

    own_offers = [o for o in offers if o.merchant_id == self.merchant_id]
    own_offers_by_uid = {offer.uid: offer for offer in own_offers}
    missing_offers = self.settings['max_amount_of_offers'] - \
        sum(offer.amount for offer in own_offers)

    start_time = time.time()
    # Buy new products if we have more room
    new_products = []
    for _ in range(missing_offers):
      try:
        prod = self.producer_api.buy_product()
        new_products.append(prod)
      except Exception as e:
        print('[Error] Could not buy new products: {}'.format(
            traceback.format_exc()))
        print('[Error] Retry in 1s')
        return 1.0

    # Get our own products from the market
    try:
      products = self.producer_api.get_products()
      product_prices_by_uid = {p.uid: p.price for p in products}
    except Exception as e:
      print('[Error] Could not get own products: {}'.format(
            traceback.format_exc()))
      print('[Error] Retry in 1s')
      return 1.0

    # Load models
    product_ids = set([p.product_id for p in products + new_products])
    models_data = model_helper.load_all_latest_models_data(
        conf['model_save_dir'],
        conf['model'],
        product_ids)
    models = {p: model for p,
              ((model, timestamp), ckpt_id) in models_data.items()}
    selectors = {}
    if conf['feature_selection']:
      model_ckpt_ids = {p: ckpt_id for p, ((model, timestamp), ckpt_id)
                        in models_data.items()}
      for p, ckpt_id in model_ckpt_ids.items():
        selector = model_helper.load_checkpoint(
            conf['selector_save_dir'], conf['model'],
            p, ckpt_id, message='selector')
        if selector is not None:
          selectors.update({p: selector[0]})
        else:
          print('[Warning] Could not load selector for {}'.format(p))
    # Set prices for already existing products
    for own_offer in own_offers:
      pid = own_offer.product_id
      model = models[pid] if pid in models else None
      if own_offer.amount > 0:
        own_offer.price = self.price_product(
            model,
            own_offer,
            product_prices_by_uid,
            current_offers=offers,
            feature_selector=selectors[pid] if pid in selectors else None)
        if self._silently_update_offer(own_offer):
          request_count += 1

    # Set prices for new products
    for product in new_products:
      pid = product.product_id
      model = models[pid] if pid in models else None
      if product.uid not in self.uid_prices:
        self.uid_prices.update({product.uid: product.price})
        print('[Debug] Initial product prices: {}'.format(self.uid_prices))
      if product.uid in own_offers_by_uid:
        # The product is already known. Only update amount and price
        offer = own_offers_by_uid[product.uid]
        offer.amount += product.amount
        offer.signature = product.signature
        try:
          # Add the amount we bought with self.producer_api.get_products()
          # to the already existing marketplace offer for that product
          self.marketplace_api.restock(
              offer.offer_id,
              amount=product.amount,
              signature=product.signature)
        except Exception as e:
          print('[Error] Could not restock an offer:', e)
        offer.price = self.price_product(
            model,
            product,
            product_prices_by_uid,
            current_offers=offers,
            feature_selector=selectors[pid] if pid in selectors else None)
        if self._silently_update_offer(offer):
          request_count += 1
      else:
        # Create a new offer for a new product and add it to the marketplace
        offer = Offer.from_product(product)
        offer.prime = True
        offer.shipping_time['standard'] = self.settings['shipping']
        offer.shipping_time['prime'] = self.settings['primeShipping']
        offer.merchant_id = self.merchant_id
        offer.price = self.price_product(
            model,
            product,
            product_prices_by_uid,
            current_offers=offers + [offer],
            new=True,
            feature_selector=selectors[pid] if pid in selectors else None)
        try:
          self.marketplace_api.add_offer(offer)
        except Exception as e:
          print('[Error] Could not add an offer to the marketplace:', e)

    # execution_time = time.time() - start_time
    # req_per_sec = max(1.0, request_count) / execution_time
    # timeout = req_per_sec / self.settings['max_req_per_sec']
    # This strange return value just pauses our thread for n seconds
    print('[Debug] Logic loop timeout set to {}'.format(
        self.settings['max_req_per_sec']))
    return self.settings['max_req_per_sec']


merchant_logic = MLMerchant()
merchant_server = MerchantServer(merchant_logic, True)
app = merchant_server.app

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='PriceWars Merchant')
  parser.add_argument(
      '--port',
      type=int,
      default=8090,
      help='Port to bind flask App to')
  args = parser.parse_args()
  app.run(host='0.0.0.0', port=args.port)
