import numpy as np


def distogram_distances(distogram_prediction):
  '''
  This function creates a distogram matrix from the logits and bin_edges of the prediction
  par distogram_prediction = dict of distogram from prediction
  '''
  bin_edges =np.concatenate(([0],distogram_prediction["bin_edges"]))

  len_log = len(distogram_prediction["logits"])
  len_log_i = len(distogram_prediction["logits"][0])

  distogram= np.empty((len_log, len_log_i), dtype=float)

  for i in range(len_log):
    for x in range(len_log_i):
      distogram[i][x] = bin_edges[np.argmax(distogram_prediction["logits"][i][x])]
      
  return distogram


