from abc import ABC, abstractmethod
import keras_tuner as kt
import keras
import numpy as np
import collections
import copy
import tensorflow as tf


class SharpeLoss(kt.keras.losses.Loss):

    def __init__(self, output_size: int = 1):
        self.output_size = output_size
        super().__init__()

    def call(self, y_true, weights):
        captured_returns = weights * y_true
        mean_returns = tf.reduce_mean(captured_returns)
        return - (
            mean_returns
            / tf.sqrt(
                tf.reduce_mean(tf.square(captured_returns))
                - tf.square(mean_returns)
                + 1e-9
            )
            * tf.sqrt(252)
        )
    

class SharpeValidationLoss(keras.callback.Callback):
    
    def __init__(
            self,
            inputs,
            returns,
            time_indices,
            num_time,
            early_stopping_patience,
            n_multiprocessing_workers,
            weights_save_location='tmp/checkpoint',
            min_delta = 1e-4
    ):
        super(keras.callbacks.Callback, self).__init__()
        self.inputs = inputs
        self.returns = returns
        self.time_indices = time_indices
        self.n_multiprocessing_workers = n_multiprocessing_workers
        self.early_stopping_patience = early_stopping_patience
        self.num_time = num_time
        self.min_delta = min_delta

        self.best_sharpe = np.NINF
        self.weights_save_location = weights_save_location

    def set_weights_save_loc(self, weights_save_location):
        self.weights_save_location = weights_save_location
    
    def on_train_begin(self, logs=None):
        self.patience_counter = 0
        self.stopped_epoch = 0
        self.best_sharpe = np.NINF

    def on_epoch_end(self, epoch, logs=None):
        positions = self.model.predict(
            self.inputs,
            workers=self.n_multiprocessing_workers,
            use_multiprocessing=True
        )

        captured_returns = tf.math.unsorted_segment_mean(
            positions * self.returns, self.time_indices, self.num_time
        )[1:]

        sharpe = (
            tf.reduce_mean(captured_returns)
            / tf.sqrt(
                tf.math.reduce_variance(captured_returns)
                + tf.constant(1e-9, dtype=tf.float64)
            )
            * tf.sqrt(tf.constant(252.0, dtype=tf.float64))
        ).numpy()
        if sharpe > self.best_sharpe + self.min_delta:
            self.best_sharpe = sharpe
            self.patience_counter = 0  # reset the count
            # self.best_weights = self.model.get_weights()
            self.model.save_weights(self.weights_save_location)
        else:
            # if self.verbose: #TODO
            self.patience_counter += 1
            if self.patience_counter >= self.early_stopping_patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                self.model.load_weights(self.weights_save_location)
        logs["sharpe"] = sharpe  # for keras tuner
        print(f"\nval_sharpe {logs['sharpe']}")


class TunerValidationLoss(kt.tuners.RandomSearch):
    def __init__(
            self,
            hypermodel,
            objective,
            max_trials,
            hp_minibatch_size,
            seed=None,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            **kwargs
    ):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs
        )
    
    def run_trial(self, trial, *args, **kwargs):

        kwargs['batch_size'] = trial.hyperparameters.Choice(
            'batch_size', values=self.hp_minibatch_size
        )
        super(TunerValidationLoss, self).run_trial(trial, *args, **kwargs)




class TunerDiversifiedSharpe(kt.tuners.RandomSearch):

    def __init__(
            self,
            hypermodel,
            objective,
            max_trials,
            hp_minibatch_size,
            seed=None,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            **kwargs
    ):
        
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(
            hypermodel,
            objective,
            max_trials,
            seed,
            hyperparameters,
            tune_new_entries,
            allow_new_entries,
            **kwargs
        )
    
    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            'batch_size', values = self.hp_minibatch_size
        )

        original_callbacks = kwargs.pop("callback", [])

        for callback in original_callbacks:
            if isinstance(callback, SharpeValidationLoss):
                callback.set_weights_save_loc(
                    self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                )

        metrics = collections.defaultdict(list)
        for execution in range(self.executions_per_trial):
            copied_fit_kwargs = copy.copy(kwargs)
            callbacks = self._deepcopy_callbacks(original_callbacks)
            self._configure_tensorboard_dir(callbacks, trial, execution)
            callbacks.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            copied_fit_kwargs['callbacks'] = callbacks

            history = self._build_and_fit_model(trial, args, copied_fit_kwargs)

            for metric, epoch_values in history.history.items():
                if self.oracle.objective.direction == 'min':
                    best_value = np.min(epoch_values)
                else:
                    best_value = np.max(epoch_values)
                metrics[metric].append(best_value)
        
        averaged_metrics = {}
        for metric, execution_values in metrics.items():
            averaged_metrics[metric] = np.mean(execution_values)
        self.oracle.update_trial(
            trial.trial_id, metrics=averaged_metrics, step=self._reported_step
        )




class DeepMomentumNetworkModel(ABC):

    def __init__(self, project_name, hp_directory, hp_minibatch_size, **params):

        params = params.copy()

        self.time_steps = int(params.pop('total_time_steps', np.nan))
        self.input_size = int(params.pop('input_size', np.nan))
        self.output_size = int(params.pop('output_size', np.nan))
        self.n_multiprocessing_workers = int(params.pop('multiprocessing_workers'), np.nan)
        self.num_epochs = int(params.pop('num_epochs', np.nan))
        self.early_stopping_patience = int(params.pop('early_stopping_patience', np.nan))
        self.random_search_iterations = int(params.pop('random_search_iterations', np.nan))
        self.evaluate_diversified_val_sharpe = int(params.pop('evaluate_diversified_val_sharpe', np.nan))
        self.force_output_sharpe_length = int(params.pop('force_output_sharpe_length', np.nan))

        if self.evaluate_diversified_val_sharpe:

            


