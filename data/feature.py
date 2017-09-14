from collections import OrderedDict
import numpy as np

import config
conf = config.get_config()


def pandas_buffer_to_product_dict(pandas_list, header):
  data = {}
  for h in header:
    data[h] = []
  for p in pandas_list:
    for h in header:
      data[h].append(p.__getattribute__(h))
  return data


def price(data, index):
  return data['price'][index]


def price_distance_to_cheapest(data, my_index):
  prices = data['price']
  min_price = min(prices)
  my_price = prices[my_index]
  distance = my_price - min_price
  return distance


def price_distance_to_nth_cheapest(data, n, my_index):
  prices = data['price']
  if n >= len(prices) or n < -len(prices):
    return 0.0
  my_price = prices[my_index]
  sorted_prices = sorted(prices)
  nth_price = sorted_prices[n]
  distance = my_price - nth_price
  return distance


def _rank(data, my_value):
  rank = 1
  for d in data:
    if d < my_value:
      rank += 1
  return rank


def quality(data, my_index):
  qualities = data['quality']
  return qualities[my_index]


def quality_rank(data, my_index):
  qualities = data['quality']
  my_quality = qualities[my_index]
  return _rank(qualities, my_quality)


def price_rank(data, my_index):
  prices = data['price']
  my_price = prices[my_index]
  return _rank(prices, my_price)


def num_competitors(data, my_index):
  merchants = data['merchant_id']
  my_id = merchants[my_index]
  merchants = set(merchants)
  competitors = 0
  for m in merchants:
    if m != my_id:
      competitors += 1
  return competitors


def average_price(data):
  prices = data['price']
  if len(prices) == 0:
    return 0.0
  return sum(prices) / len(prices)


def product_id(data, my_index):
  product_ids = data['product_id']
  return product_ids[my_index]


def offer_id(data, my_index):
  offer_ids = data['offer_id']
  return offer_ids[my_index]


def extract_features_fn(data, my_index):
  """Expects a dict where each key links to a list of values.
  One of the keys has to be 'merchant_id' which will be used to
  identify own offers within the lists.

  Args:
    data: A product_dict from e.g. pandas_buffer_to_product_dict().
    my_index: Used to identify the column with own products.

  Returns:
    A dictionary with the computed features, label and product_id as keys.

  """
  p = price(data, my_index)
  p_rank = price_rank(data, my_index)
  # p_avg = average_price(data)
  data = {
      'features': OrderedDict([
          ('own_price', p),
          ('own_price_2', pow(p, 2)),
          ('price_rank', p_rank),
          ('price_rank_2', pow(p_rank, 2)),
          # ('price_rank_3', pow(p_rank, 3)),
          ('price_distance_to_cheapest', price_distance_to_cheapest(data,
                                                                    my_index)),
          # ('price_distance_to_2nd_cheapest', price_distance_to_nth_cheapest(
              # data, 1, my_index)),
          # ('price_distance_to_3ed_cheapest', price_distance_to_nth_cheapest(
              # data, 2, my_index)),
          # ('average_price', p_avg),
          # ('price_distance_to_average', p - p_avg),
          # ('price_standard_deviation', np.std(data['price'])),
          # ('price_variance', np.var(data['price'])),
          ('quality', quality(data, my_index)),
          ('quality_rank', quality_rank(data, my_index))
          # ('num_competitors', num_competitors(data, my_index)),
      ]),
      # Whether we sold or not
      'label': 0,
      'product_id': product_id(data, my_index),
      'offer_id': offer_id(data, my_index)
  }
  return data


def compute_situations_dict(
        market_situations,
        merchant_id,
        buy_offers=None,
        feature_fn=extract_features_fn,
        binary_label=True):
  """Combines market_situations and buy_offers and computes the features.

  Args:
    market_situations: Result of csvparser.read_csv
    merchant_id: The id used to identify own products
    buy_offers: Result of csvparser.read_csv. If None, then label will be set
      to 0 for every timestamp.
    feature_fn: A function fn(data, (merchant_id, my_index)) that returns
      computed features. The returned data structure must support access by
      string index e.g. data['label']. It has to contain at least the following
      keys: features, label, product_id, offer_id
    binary_label: If False label will be set to the number of items sold. If
      True it will set 1 for a sales event otherwise 0

  Returns:
    An OrderedDict where the key is a tupel of (timestamp, offer_id) and the
    value is the result of a feature_fn.
  """
  situations = OrderedDict()
  buff = []
  last_timestamp = market_situations.iloc[0].timestamp

  used_headers = [
      h for h in market_situations.columns if h not in conf['ignored_header']]
  for row in market_situations.itertuples():
    if row.timestamp == last_timestamp:
      buff.append(row)
    else:
      data = pandas_buffer_to_product_dict(buff, used_headers)
      indices = [i for i, x in enumerate(data['merchant_id']) if
                 x == merchant_id]
      # Compute features for each of our products in the situation
      for i in indices:
        features = feature_fn(data, i)
        # Only take situations where we actually had a product
        if features is not None:
          situations[(last_timestamp, features['offer_id'])] = features
      buff = []
      buff.append(row)
      last_timestamp = row.timestamp
  if len(buff) > 0:
    data = pandas_buffer_to_product_dict(buff, used_headers)
    # Find all our products in the current market situation
    indices = [i for i, x in enumerate(data['merchant_id']) if
               x == merchant_id]
    # Compute features for each of our products in the situation
    for i in indices:
      features = feature_fn(data, i)
      # Only take situations where we actually had a product
      if features is not None:
        situations[(last_timestamp, features['offer_id'])] = features
  # Compute some utility variables
  timestamps = []
  offer_ids = []
  keys = []
  for v in situations.keys():
    timestamps.append(v[0])
    offer_ids.append(v[1])
    keys.append(v)
  if buy_offers is not None:
    # Go through all buy offers one by one
    buy_offer_index = 0
    timestamp_index = 0
    while (buy_offer_index < len(buy_offers) and
           timestamp_index < len(timestamps)):
      # Find the range of possible timestamps in the market situation
      # that match the buy offer timestamp

      # Move the lower_bound up if the new row is later in time than the
      # previous upper_bound
      current_buy_offer = buy_offers.loc[buy_offer_index]
      # Market situation timestamp
      current_timestamp = timestamps[timestamp_index]
      if current_timestamp < current_buy_offer.timestamp:
        timestamp_index += 1
        continue

      sale_index = timestamp_index - 1
      if current_timestamp == current_buy_offer.timestamp:
        sale_index = i
      while current_buy_offer.offer_id != offer_ids[sale_index]:
        sale_index -= 1
      if current_buy_offer.offer_id == offer_ids[sale_index]:
        if binary_label:
          situations[keys[sale_index]]['label'] = 1
        else:
          situations[keys[sale_index]]['label'] += current_buy_offer.amount
      buy_offer_index += 1
    while buy_offer_index < len(buy_offers):
      current_buy_offer = buy_offers.loc[buy_offer_index]
      if current_buy_offer.offer_id == offer_ids[timestamp_index - 1]:
        if binary_label:
          situations[keys[i]]['label'] = 1
        else:
          situations[keys[i]]['label'] += row.amount
      buy_offer_index += 1
  return situations


def get_list_of_features_from_situations_dict(situations_dict):
  return [list(v['features'].values()) for v in situations_dict.values()]
