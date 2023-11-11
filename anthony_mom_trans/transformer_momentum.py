import tensorflow as tf
from tensorflow import keras
import gc
import numpy as np
from copy import deepcopy
# Please change to your own function eventually.
from mom_trans.deep_momentum_network import DeepMomentumNetworkModel, SharpeLoss
from settings.hp_grid import (
    HP_DROPOUT_RATE,
    HP_HIDDEN_LAYER_SIZE,
    HP_LEARNING_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_MINIBATCH_SIZE,
)

def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    """Returns simple Keras linear layer.
    Parameters
    ----------
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = keras.layers.Dense(size, activation=None, use_time_distributed=False, use_bias=True)
    if use_time_distributed:
        linear = keras.layers.TimeDistributed(linear)
    return linear

def apply_mlp(
        inputs,
        hidden_size,
        output_size,
        output_activation=None,
        hidden_activation='tanh',
        use_time_distributed=False
):
    """Applies simple feed-forward network to an input.
    Args:
      inputs: MLP inputs
      hidden_size: Hidden state size
      output_size: Output size of MLP
      output_activation: Activation function to apply on output
      hidden_activation: Activation function to apply on input
      use_time_distributed: Whether to apply across time
    Returns:
      Tensor for MLP outputs.
    """
    if use_time_distributed:
        hidden = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_size, activation=hidden_activation)
        )(inputs)
        return keras.layers.TimeDistributed(
            keras.layers.Dense(output_size, activation=output_activation)
        )
    else:
        hidden = keras.layers.Dense(hidden_size, activation=hidden_activation)(inputs)
        return keras.layers.Dense(output_size, activation=output_activation)(hidden)

def apply_gating_layer(
        x,
        hidden_layer_size: int,
        dropout_rate: float = None,
        use_time_distributed: bool = True,
        activation = None
):
    """Applies a Gated Linear Unit (GLU) to an input.
    Parameters
    ----------
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature transform if necessary
    Returns
    -------
        Tuple of tensors for: (GLU output, gate)
    """

    if dropout_rate:
        x = keras.layers.Dropout(dropout_rate)(x)


    if use_time_distributed:
        activation_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_layer_size, activation=activation)
        )(x)
        gated_layer = keras.layers.TimeDistributed(
            keras.layers.Dense(hidden_layer_size, activation='sigmoid')
        )(x)
    else:
        activation_layer = keras.layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = keras.layers.Dense(hidden_layer_size, activation='sigmoid')(x)
    
    return keras.layers.multiply([activation_layer, gated_layer]), gated_layer

def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.
    Parameters
    ----------
        x_list: List of inputs to sum for skip connection
    Returns
    -------
        Tensor output from layer.
    """
    tmp = keras.layers.Add()(x_list)
    tmp = keras.layers.LayerNormalization()(tmp)
    return tmp


def gated_residual_network(
        x,
        hidden_layer_size: int,
        output_size: int = None,
        dropout_rate: float = None,
        use_time_distributed: bool = True,
        additional_context = None,
        return_gate: bool = False
        ):
    """Applies the gated residual network (GRN) as defined in paper.
    Parameters
    ----------
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes
    Returns
    -------
        Tuple of tensors for: (GRN output, GLU gate)
    """

    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = keras.layers.Dense(output_size)
        if use_time_distributed:
            linear = keras.layers.TimeDistributed(linear)
        skip = linear(x)

    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(x)
    if additional_context:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False
        )(additional_context)
    hidden = keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size, activation=None, use_time_distributed=use_time_distributed
    )(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None
    )

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.
    Parameters
    ----------
        self_attn_inputs: Inputs to self attention layer to determine mask shape
    """
    len_s = tf.shape(self_attn_inputs)[-2]
    bs = tf.shape(self_attn_inputs)[:-2]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), -2)
    return mask
    

class ScaledDotProductAttention(keras.layers.Layer):
    """
    Defines scaled dot producrt attention layer.
    Attributes
    ----------

    dropout: Dropout rate to use
    activation: normalisation function for scaled dot product attention (e.g. softmax by default)
    """

    def __init__(self, attn_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = keras.layers.Dropout(attn_dropout)
        self.activation = keras.layers.Activation("softmax")

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention,
        
        Parameters
        ----------
            q: queries
            k: keys
            v: values
            mask: Masking if required - sets softmax to very large value
        Returns
        ----------
            tuple of (layer outputs, attention weights)

        """
        attn = keras.layers.Lambda(self.scaled_batchdot)([q, k])
        if mask is not None:
            mmask = keras.layers.Lambda(lambda x: (-1e9) * (1.0 - tf.cast(x, 'float32')))(mask)
            attn = keras.layers.add([attn, mmask])
        attn = self.activate(attn)
        attn = self.dropout(attn)
        output = keras.layers.Lambda(lambda x: keras.backend.batch_dot(x[0], x[1]))([attn, v])
        return output, attn
    
    def scaled_batchdot(input_list):
        d, k = input_list
        dimension = tf.sqrt(tf.cast(k.shape[-1], dtype='float32'))
        return keras.backend.batch_dot(d, k, axes=[2, 2]) / dimension

class InterpretableMultiHeadAttention(keras.layers.Layer):
    """Defines interpretable multi-head attention layer.
    Attributes:
        n_head: Number of heads
        d_k: Key/query dimensionality per head
        d_v: Value dimensionality
        dropout: Dropout rate to apply
        qs_layers: List of queries across heads
        ks_layers: List of keys across heads
        vs_layers: List of values across heads
        attention: Scaled dot product attention layer
        w_o: Output weight matrix to project internal state to the original TFT state size
    """

    def __init__(self, n_head: int, d_model: int, dropout: float, **kwargs):
        """Initialises layer.
        Parameters
        ----------
            n_head: Number of heads
            d_model: TFT state dimensionality
            dropout: Dropout discard rate
        """

        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout
        
        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        vs_layer = keras.layers.Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(keras.layers.Dense(d_k, use_bias=False))
            self.ks_layers.append(keras.layers.Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)

        self.attention = ScaledDotProductAttention()  # Maybe change _this_ class to inherit?
        self.w_o = keras.layers.Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.
        Using T to denote the number of time steps fed into the transformer.
        Parameters
        ----------
            q: Query tensor of shape=(?, T, d_model)
            k: Key of shape=(?, T, d_model)
            v: Values of shape=(?, T, d_model)
            mask: Masking if required with shape=(?, T, T)
        Returns
        -------
            Tuple of (layer outputs, attention weights)
        """

        heads = []
        attns = []

        for i in range(self.n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = keras.layers.Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)

        keras.backend.stack((lambda x, axis=0: [x] if not isinstance(x, list) else x), axis=0)
        head = keras.layers.Lambda(
            keras.backend.stack(
                (lambda x, axis=0: [x] if not isinstance(x, list) else x), axis=0)
                )(heads) if self.n_head >= 1 else heads[0]
        attn = keras.layers.Lambda(
            keras.backend.stack(
                (lambda x, axis=0: [x] if not isinstance(x, list) else x), axis=0)
        )(attns)

        outputs = keras.layers.Lambda(keras.backend.mean, arguments={"axis": 0})(head) if self.n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = keras.layers.Dropout(self.dropout)(outputs)

        return outputs, attn


class TFTDeepMomentumNetworkModel(DeepMomentumNetworkModel):

    def __init__(
            self, 
            project_name, 
            hp_directory, 
            hp_minibatch_size=HP_MINIBATCH_SIZE,
            **params
        ):

        copy_params = deepcopy(params)
        self._input_placeholder = None
        self._attention_components = None
        self._prediction_parts = None

        self._static_input_loc = params.pop('static_input_loc', np.nan)
        self._known_regular_input_idx = params.pop('known_regular_inputs', np.nan)
        self._known_categorical_input_idx = params.pop('known_categorical_inputs', np.nan)
        self.category_counts = params.pop('category_counts', np.nan)

        self.column_definition = params.pop('column_definition', np.nan)

        self.num_encoder_steps = params.pop('num_encoder_steps', np.nan)
        self.num_stacks = params.pop('stack_size', np.nan)
        self.num_heads = params.pop('num_heads', np.nan)
        self.input_size = int(params.pop('input_size', np.nan))
        super().__init__(project_name, hp_directory, hp_minibatch_size, **copy_params)

    def model_builder(self, hp):
        pass

    def get_tft_embeddings(self, all_inputs):
        """Transforms raw inputs into embeddings.
        
        Applies a linear transformation onto continuous variables 
        and uses embeddings for categorical variables
        
        Parameters
        ----------  
            all_inputs: inputs to transform
        
        Returns
        ---------
            tensor for transformer inputs
        
        """

        time_steps = self.time_steps

        if all_inputs.get_shape().as_list()[-1] != self.input_size:
            raise ValueError(
                f"Illegal number of inputs!\
                Observed={all_inputs.get_shape().as_list()[-1]}.\
                Expected={self.input_size}."
            )
    
        num_categorical_variables = len(self.category_counts)
        num_regular_variables = self.input_size - num_categorical_variables

        embedding_sizes = [
            self.hidden_layer_size for _, _ in enumerate(self.category_counts)
        ]


        embeddings = []
        for i in range(num_categorical_variables):

            embedding = keras.Sequential(
                [
                    keras.layers.InputLayer([time_steps]),
                    keras.layers.Embedding(
                        self.category_counts[i],
                        embedding_sizes[i],
                        input_length=time_steps,
                        dtype=tf.float32
                    )
                ]
            )
            embeddings.append(embedding)

        regular_inputs, categorical_inputs = (
            all_inputs[:, :, :num_regular_variables],
            all_inputs[:, :, num_regular_variables:]
        )

        embedded_inputs = [
            embeddings[i](categorical_inputs[Ellipsis, i])
            for i in range(num_categorical_variables)
        ]

        if self._static_input_loc:
            static_inputs = [
                keras.layers.Dense(self.hidden_layer_size)(
                    regular_inputs[:, 0, i:i+1]
                )
                for i in range(num_regular_variables)
                if i in self._static_input_loc
            ] + [
                embedded_inputs[i][:, 0, :]
                for i in range(num_categorical_variables)
                if i + num_regular_variables in self._static_input_loc
            ]
            static_inputs = keras.backend.stack(static_inputs, axis=1)
        
        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            return keras.layers.TimeDistributed(
                keras.layers.Dense(self.hidden_layer_size)
            )
        
        wired_embeddings = []
        for i in range(num_categorical_variables):
            if i not in self._known_categorical_input_idx:
                e = embeddings[i](categorical_inputs[:, :, i])
                wired_embeddings.append(e)

        unknown_inputs = []
        for i in range(regular_inputs.shape[-1]):
            if i not in self._known_regular_input_idx:
                e = convert_real_to_embedding(regular_inputs[Ellipsis, i : i + 1])
                unknown_inputs.append(e)
        
        if unknown_inputs + wired_embeddings:
            unknown_inputs = keras.backend.stack(
                unknown_inputs + wired_embeddings, axis=-1
            )
        else:
            unknown_inputs = None
        
        known_regular_inputs = [
            convert_real_to_embedding(regular_inputs[Ellipsis, i:i+1])
            for i in self._known_regular_input_idx
            if i not in self._static_input_loc
        ]

        known_categorical_inputs = [
            embedded_inputs[i]
            for i in self._known_categorical_input_idx
            if i + num_regular_variables not in self._static_input_loc
        ]

        known_combined_layer = keras.backend.stack(
            known_regular_inputs + known_categorical_inputs, axis=-1
        )

        return unknown_inputs, known_combined_layer, static_inputs
    
    def get_attention(self, data, batch_size, mask=None):

        if mask:
            inputs = data["inputs"][mask]
            identifiers = data['identifier'][mask]
            time = data["date"][mask]

        else:
            inputs = data["inputs"]
            identifiers = data['identifier']
            time = data['date']

        def get_batch_attention_weights(input_batch):

            input_placeholder = self._input_placeholder
            attention_weights = {}

            for k in self._attention_components:
                extractor = tf.keras.Model(
                    inputs=input_placeholder, outputs=self._attention_components[k]
                )
                attention_weight = extractor(input_batch.astype(np.float32))
                attention_weights[k] = attention_weight
            
            return attention_weights
        
        n = inputs.shape[0]
        num_batches = n // batch_size
        if n - (num_batches * batch_size) > 0:

            num_batches += 1

        batched_inputs = [
            inputs[i * batch_size: (i+1) * batch_size, Ellipsis]
            for i in range(num_batches)
        ]

        attention_by_batch = [
            get_batch_attention_weights(batch) for batch in batched_inputs
        ]

        attention_weights = {}
        for k in self._attention_components:
            attention_weights[k] = []
            for batch_weights in attention_by_batch:
                attention_weights[k].append(batch_weights[k])
            
            if len(attention_weights[k][0].shape) == 4:
                tmp = np.concatenate(attention_weights[k], axis=1)
            else:
                tmp = np.concatenate(attention_weights[k], axis=0)

            del attention_weights[k]
            gc.collect()
            attention_weights[k] = tmp

        attention_weights['identifiers'] = identifiers[:, 0, 0]
        attention_weights['time'] = time[:, :, 0]

        return attention_weights
        





