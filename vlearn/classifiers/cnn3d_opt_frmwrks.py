import pdb
import numpy as np
from .cnn3d_architectures import CNN3DArchs
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import tensorflow as tf


class OptFrmWrk(CNN3DArchs):
    """
    Optimization frameworks to explore and find best models.

    Todo:
        * Use singel cv inside nested cv
    """

    def __init__(self, Xtr, ytr):
        """
        Args:
            X (np_array): A numpy array having training samples
            y (np._array): A numpy array having labels
        """
        self._Xtr = Xtr
        self._ytr = ytr

    def nested_cv(self, params, split):
        """
        Optimizes for best parameters and model using nested cv.
        
        Args:
            params (dict): Dictionary of parameters to optimize
            split (tuple): A tuple having cross validation split parts.
                (inner split, outer split)
        """
        var_param, stat_param = self._get_var_params(params)
        param_grid = ParameterGrid(var_param)
        
        in_cv = StratifiedKFold(split[0])
        out_cv = StratifiedKFold(split[1])

        # Outer cross validation loop
        best_perfs = []
        best_params_lst = []
        for out_tr_idx, out_tst_idx in out_cv.split(self._Xtr, self._ytr):
            print("Outer CV")

            # Parameter loop
            param_best_perf = -np.inf
            for pidx, cparams in enumerate(param_grid):
                print("\tParameters loop")

                # Build model based on parameters
                all_cparams = {**cparams, **stat_param}
                model = CNN3DArchs(all_cparams,
                                   self._Xtr[out_tr_idx],
                                   self._ytr[out_tr_idx]).build_model()
                epochs_ = all_cparams["epochs"]
                batch_size_ = all_cparams["batch_size"]

                # Inner cross validation loop
                in_perfs = []
                for in_tr_idx, in_tst_idx in in_cv.split(self._Xtr[out_tr_idx], self._ytr[out_tr_idx]):
                    tf.keras.backend.clear_session()
                    model.fit(
                        self._Xtr[in_tr_idx],
                        self._ytr[in_tr_idx],
                        epochs=epochs_,
                        validation_split=0.2,
                        batch_size=batch_size_,
                        verbose=0,
                    )
                    in_loss, in_perf = model.evaluate(
                        self._Xtr[in_tst_idx], self._ytr[in_tst_idx], verbose=0
                    )
                    in_perfs.append(in_perf)
                    print("\t\tInner CV ", str(in_perf))

                # Mean inner performance
                in_mean_perf = np.mean(in_perfs)
                print("\t\tMean performance ", str(in_mean_perf))
                if in_mean_perf > param_best_perf:
                    param_best_perf = in_mean_perf
                    best_params = cparams

            print("\tInner best parameters ", best_params)
            print("\tMean Best performance ", param_best_perf)

            # Performance of best parameters on outer split
            all_cparams = {**best_params, **stat_param}
            model = CNN3DArchs(all_cparams, self._Xtr[out_tr_idx], self._ytr[out_tr_idx]).build_model()
            epochs_ = all_cparams["epochs"]
            batch_size_ = all_cparams["batch_size"]
            tf.keras.backend.clear_session()
            model.fit(
                self._Xtr[out_tr_idx],
                self._ytr[out_tr_idx],
                epochs=epochs_,
                validation_split=0.2,
                batch_size=batch_size_,
                verbose=0,
            )
            out_loss, out_perf = model.evaluate(self._Xtr[out_tst_idx], self._ytr[out_tst_idx], verbose=0)
            print("Best parameters ", best_params)
            print("Performance on outer testing ", str(out_perf))

            # Storing best parameters for outer loop
            best_params_lst.append({**stat_param,**best_params})
            best_perfs.append(out_perf)
        return best_params_lst, best_perfs

    def _get_var_params(self, params):
        """
        Returns parameters that can be varied during optimization.
        Args:
            params (dict): All the parameters in an array
        """
        var_dict = {}
        static_dict = {}
        for key in params:
            if len(params[key]) > 1:
                var_dict[key] = params[key]
            else:
                static_dict[key] = params[key][0]

        return var_dict, static_dict
