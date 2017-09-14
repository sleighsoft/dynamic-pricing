import pickle
import pandas as pd


def read_csv(path_or_url):
  """Loads a csv from local filesystem or url into a pandas dataframe."""
  return pd.read_csv(path_or_url, parse_dates=['timestamp'])


def save_situations_to_disk(situations, path):
  """Saves situations as a pickle file to disk"""
  with open(path, 'wb') as f:
    pickle.dump(situations, f)


def load_situations_from_disk(path):
  """Loads situations from a pickle file"""
  with open(path, 'rb') as f:
    return pickle.load(f)
