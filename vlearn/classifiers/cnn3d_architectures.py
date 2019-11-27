import sys
import pdb
import tensorflow as tf
import tensorflow.keras.layers as tfkr_layers


class CNN3DArchs:
    """
    Contains methods to build 3D CNN architectures. The architecture is a tf.keras
    model and supports funtional api of tf.keras.
    """

    def __init__(self, params, X, y):
        """
        Returns a tf.keras model
        """
        self._X = X
        self._y = y
        self._params = params

    def build_model(self):
        """
        Builds a tf.keras model based the parameters.
        
        Args:
            params (dict):
                Parameters to use for building model.
        """
        arch_name = self._params["arch_name"]
        if arch_name == "flat":
            model = self._build_flat_model()
        else:
            print("Architecture not supported ", arch_name)
            sys.exit()
        return model

    def _build_flat_model(self):
        """
        Builds a flat cnn3d model using tensorflow 2. It has same
        number of convolutional kernels throughout.
        Args:
            params (dict): Dictionary having parameters for architecture.
                1. num_conv_layers
                2. num_kernels
        """
        # Extracting architecture parameters from dictionary
        num_conv_layers_ = self._params["num_conv_layers"]
        num_kernels_ = self._params["num_kernels"]
        kernel_size_ = self._params["kernel_size"]
        activation_ = self._params["activation"]
        data_format_ = self._params["data_format"]
        pool_size_ = self._params["pool_size"]
        batch_norm_ = self._params["batch_norm"]
        num_dense_layers_ = self._params["num_dense_layers"]
        final_activation_ = self._params["final_activation"]
        loss_ = self._params["loss"]
        optimizer_ = self._params["optimizer"]
        dense_units_ = self._params["dense_units"]
        metric_ = self._params["metric"]

        # Input Layer
        sample_shape = self._X.shape[1:]
        input_layer = tfkr_layers.Input(sample_shape)

        # First convoluton and pooling layers
        conv_layer = tfkr_layers.Conv3D(
            filters=num_kernels_,
            kernel_size=kernel_size_,
            activation=activation_,
            data_format=data_format_,
        )(input_layer)
        pool_layer = tfkr_layers.MaxPool3D(
            pool_size=pool_size_, data_format=data_format_
        )(conv_layer)

        # Remaining convolution and pooling layers
        for layer_idx in range(1, num_conv_layers_):
            conv_layer = tfkr_layers.Conv3D(
                filters=num_kernels_,
                kernel_size=kernel_size_,
                activation=activation_,
                data_format=data_format_,
            )(pool_layer)
            pool_layer = tfkr_layers.MaxPool3D(
                pool_size=pool_size_, data_format=data_format_
            )(conv_layer)

        # Batch Normalization
        if batch_norm_:
            pool_layer = tfkr_layers.BatchNormalization()(pool_layer)

        # Flatten
        flat_layer = tfkr_layers.Flatten()(pool_layer)

        # dense layers
        dense_layer = tfkr_layers.Dense(units=dense_units_, activation=activation_)(
            flat_layer
        )
        for layer_idx in range(1, num_dense_layers_):
            dense_layer = tfkr_layers.Dense(units=dense_units_, activation=activation_)(
                dense_layer
            )

        # output layer, sigmoid for binary classificaiton
        output_layer = tfkr_layers.Dense(units=1, activation=final_activation_)(
            dense_layer
        )

        # Return model
        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss=loss_, optimizer=optimizer_, metrics=[metric_])

        return model
