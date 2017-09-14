from sklearn.metrics import precision_score, recall_score
import numpy as np


def aic(log_likelihood, number_of_features):
  """Larger is better """
  return -2 * log_likelihood + 2 * number_of_features


def bic(log_likelihood, number_of_features, data_size):
  """Larger is better """
  return -2 * log_likelihood + data_size * number_of_features


def mc_fadden(log_likelihood, null_model_log_likelihood):
  """Closer to 1 is better."""
  return 1 - log_likelihood / null_model_log_likelihood


def null_model_log_like(labels):
  """Calculates the log_likelihood for a model that always predicts 1.
  Requires labels to contain exactly 2 classes [0, 1].
  """
  avg_prob = sum(labels) / len(labels)
  probabilities = np.array([[1 - avg_prob, avg_prob]] * len(labels))
  return log_like(probabilities, labels)


def log_like(y_predictions, y_true):
  """Calculates the log likelihood."""
  y_pred = y_predictions[:, 1]
  classes = len(set(y_true))
  if classes > 2:
    raise ValueError('Expected not more than 2 classes but got {}'.format(
        classes))

  epsilon = 0.00001
  ones = np.full(len(y_pred), 1)
  return sum(y_true * np.log(y_pred + epsilon) + (ones - y_true) *
             np.log(ones - y_pred + epsilon))


def print_all_scores(labels, predictions, prediction_probabilities,
                     number_of_features, data_size, print_scores=False):
  """Calculates, prints and returns the following scores:
  - Log-Likelihood
  - AIC
  - BIC
  - McFadden
  - Precision
  - Recall
  """
  null_model_log_likelihood = null_model_log_like(labels)
  log_likelihood = log_like(prediction_probabilities, labels)
  AIC = aic(log_likelihood, number_of_features)
  BIC = bic(log_likelihood, number_of_features, data_size)
  MC_FADDEN = mc_fadden(log_likelihood, null_model_log_likelihood)
  PRECISION = precision_score(labels, predictions)
  RECALL = recall_score(labels, predictions)
  if print_scores:
    print('Log-Likelihood: {}'.format(log_likelihood))
    print('AIC: {}'.format(AIC))
    print('BIC: {}'.format(BIC))
    print('McFadden: {}'.format(MC_FADDEN))
    print('Precision: {}'.format(PRECISION))
    print('Recall: {}'.format(RECALL))
  return log_likelihood, AIC, BIC, MC_FADDEN, PRECISION, RECALL
