from imblearn.over_sampling import RandomOverSampler, SMOTE


def random_oversample(X, Y, ratio):
  ros = RandomOverSampler(ratio=ratio)
  X_resampled, Y_resampled = ros.fit_sample(X, Y)
  return X_resampled, Y_resampled


def smote_oversample(X, Y, ratio):
  smo = SMOTE(ratio=ratio)
  X_resampled, Y_resampled = smo.fit_sample(X, Y)
  return X_resampled, Y_resampled


def sample_data(X, Y, mode, ratio):
  if mode == 'smote':
    return smote_oversample(X, Y, ratio)
  if mode == 'random':
    return random_oversample(X, Y, ratio)
  else:
    return X, Y
