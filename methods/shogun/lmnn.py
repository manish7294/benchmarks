'''
  file lmnn.py
  @author Manish Kumar

  Large Margin Nearest Neighbors with shogun.
'''

import os
import sys
import inspect
import timeout_decorator

# Import the util path, this method even works if the path contains symlinks to
# modules.
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(
  os.path.split(inspect.getfile(inspect.currentframe()))[0], "../../util")))
if cmd_subfolder not in sys.path:
  sys.path.insert(0, cmd_subfolder)

#Import the metrics definitions path.
metrics_folder = os.path.realpath(os.path.abspath(os.path.join(
  os.path.split(inspect.getfile(inspect.currentframe()))[0], "../metrics")))
if metrics_folder not in sys.path:
  sys.path.insert(0, metrics_folder)

from log import *
from timer import *
from definitions import *
from misc import *

import numpy as np
from modshogun import RealFeatures
from modshogun import MulticlassLabels, MulticlassAccuracy
from modshogun import LMNN as ShogunLMNN
from modshogun import KNN, KNN_COVER_TREE, EuclideanDistance

'''
This class implements the Large Margin Nearest Neighbors benchmark.
'''
class LMNN(object):

  '''
  Create the Large Margin Nearest Neighbors instance.

  @param dataset - Input dataset to perform LMNN on.
  @param timeout - The time until the timeout. Default no timeout.
  @param verbose - Display informational messages.
  '''
  def __init__(self, dataset, timeout=0, verbose=True):
    self.verbose = verbose
    self.dataset = dataset
    self.timeout = timeout

  '''
  Use the shogun libary to implement Large Margin Nearest Neighbors.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def LMNNShogun(self, options):
    @timeout_decorator.timeout(self.timeout)
    def RunLMNNShogun():
      totalTimer = Timer()

      # Load input dataset.
      Log.Info("Loading dataset", self.verbose)
      # Use the last row of the training set as the responses.
      X, y = SplitTrainData(self.dataset)
      try:
        feat = RealFeatures(X.T)
        labels = MulticlassLabels(y.astype(np.float64))

        with totalTimer:
          # Get the options for running LMNN.
          if "k" in options:
            k = int(options.pop("k"))
          else:
            k = 1

          if "maxiter" in options:
            n = int(options.pop("maxiter"))
          else:
            n = 2000

          if len(options) > 0:
            Log.Fatal("Unknown parameters: " + str(options))
            raise Exception("unknown parameters")

          # Perform LMNN.
          prep = ShogunLMNN(feat, labels, k)
          prep.set_maxiter(n)
          prep.train()
      except Exception as e:
        return -1

      time = totalTimer.ElapsedTime()

      # Predict labels.
      distance = prep.get_linear_transform()
      transformedData = np.dot(X, distance.T)
      feat  = RealFeatures(transformedData.T)
      labels = MulticlassLabels(y.astype(np.float64))
      dist = EuclideanDistance()
      knn = KNN(1, dist, labels)
      if "k" in options:
        knn.set_k(options.pop("k"))

      knn.train(feat)
      knn.set_knn_solver_type(KNN_COVER_TREE)
      pred = knn.apply_multiclass(feat)
      evaluator = MulticlassAccuracy()
      accuracy = evaluator.evaluate(pred, labels)
      print(accuracy)
      return [time, accuracy]

    try:
      return RunLMNNShogun()
    except timeout_decorator.TimeoutError:
      return -1

  '''
  Perform Large Margin Nearest Neighbors. If the method has been successfully
  completed return the elapsed time in seconds.

  @param options - Extra options for the method.
  @return - Elapsed time in seconds or a negative value if the method was not
  successful.
  '''
  def RunMetrics(self, options):
    Log.Info("Perform LMNN.", self.verbose)

    results = self.LMNNShogun(options)
    if results[0] < 0:
      return results[0]

    # Datastructure to store the results.
    metrics = {}
    metrics['Runtime'] = results[0]
    metrics['Avg Accuracy'] = results[1]

    return metrics

