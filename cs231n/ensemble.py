import numpy as np
from cs231n import optim
from time import time

class Ensemble(object):
  """
  An Ensemble encapsulates all the logic necessary for ensembling pre-trained
  models to perform better classification. Ensembling is a general term for 
  combining many classifiers by averaging or voting. 

  To put together an ensemble, you will first construct an Ensemble instance, 
  then pass in the trained models. A trained model consist of:
    (1) A model object that conforms to the API as defined in the Solver.
    (2) Parameters from previous training (stored in NPZ files)
  
  Example usage might look something like this:
  
  models = {'model1': MyAwesomeModel1(hidden_size=100, reg=10),
    model2': MyAwesomeModel2(hidden_size=100, reg=10)}
    
  param_files = {'file1': 'bestparams-1.npz',
    file2': 'bestparams-2.npz'}

  ensemble = Ensemble(models, param_files)

  The ensemble accepts test data and use the ensembled models to generate scores
  or check accuracy:
  
  score = ensemble.predict(test_data)
  accuracy = ensemble.check_accuracy(X_test, y_test)

  An Ensemble works on a model object 

  """

  def __init__(self, models=[], param_files=[],use_batchnorm=[],num_classes=10):
    """
    Construct a new Ensemble instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - param_files: A npz file containing parameters from trained model
      
    """
    self.models = models
    self.param_files = param_files
    self.batchnorm = use_batchnorm 
    self.num_classes = num_classes
   


  def check_accuracy(self, X, y):
    """
    Check accuracy of the ensemble on the provided data.
    
    Inputs:
    - X: Array of data, of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,)
     
    Returns:
    - acc: Scalar giving the fraction of instances that were correctly
      classified by the ensemble.
    """
    models = self.models
    param_files = self.param_files
    num_classes = self.num_classes
    num_models = len(models)
    N = len(y)
    scores_sum = np.zeros((N,num_classes))
    print "There are %d models in the Ensemble." % len(models)
    
    for n in range(num_models):
        # Load trained parameters into model
        outfile = param_files[n]
        npzfile = np.load(outfile)
        models[n].params = npzfile['params'].item()
        models[n].bn_params = npzfile['bn_params'].item()
        npzfile.close()
        
        scores = models[n].loss(X)
        print 'Model %d Test set accuracy: %f' % (n+1,(np.argmax(scores, axis=1) == y).mean())
        
        scores_sum += scores
        
    y_test_pred = np.argmax(scores_sum/n, axis=1)
    acc = (y_test_pred == y).mean()
    return acc
