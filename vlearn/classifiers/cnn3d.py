import os
import sys
import pdb
import numpy as np
import pandas as pd
from .cnn3d_opt_frmwrks import OptFrmWrk
from .cnn3d_architectures import CNN3DArchs
from sklearn.model_selection import ParameterGrid, StratifiedKFold

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class CNN3D(OptFrmWrk):
    """
    The following class provides an intuitive way to
        build custom neural networks using tensorflow 2
        for activity detection in trimmed videos.

    Todo:
        * Divide parameters into static and dynamic in this class
        * After nested cross validation use n fold cross validation to get
            best of the best model.
    """

    def __init__(self, arch_params, training_params):
        """
        Initializes ParameterGrid with different parameters that can be
        varied in architecture and training as proved in the arguments.
        
        args:
            arch_params:  Parameters that define architecture.
            train_params: Training parameter dictionary.
        """
        self._arch_params = arch_params
        self._training_params = training_params

    def get_best_model(self, Xtr, ytr, method="nestedcv", ncv_split=(3, 3)):
        """
        Optimizes for best parameters and model using nested corss validation.
        
        Args:
            Xtr (nparray): An array having samples for training
            ytr (nparray): An array having labels corresponding to each sample in Xtr
            method (str) : A string having name of the parameter
                           parameter tuning method. Default is nested cross validation.
            ncv_split (tuple): Cross validation split for nestedcv, 
                                (inner split, outer split). Default is (3,3)
        """
        # Getting optimal architecture and training parameters
        opt = OptFrmWrk(Xtr, ytr)
        if method == "nestedcv":

            params = {**self._arch_params, **self._training_params}
            ncv_best_params, perfs = opt.nested_cv(params, (3, 3))
            
            best_params = ncv_best_params[np.argmax(perfs)]
            best_model  = CNN3DArchs(best_params, Xtr, ytr).build_model()

            epochs_ = best_params["epochs"]
            batch_size_ = best_params["batch_size"]
            best_model.fit(Xtr, ytr, epochs = epochs_, batch_size = batch_size_,
                           validation_split=0.2, verbose=1)
            
        else:
            print("Parameter tuning not supported")
            sys.exit()

        return best_params, best_model
        
