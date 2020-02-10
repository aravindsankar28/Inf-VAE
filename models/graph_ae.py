import numpy as np

import tensorflow as tf
from models.layers import CustomDense, GraphConvolution, InnerProductDecoder, get_layer_uid
from utils.preprocess import normalize_graph_gcn, sparse_to_tuple

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class GraphAE(object):
    """Base GraphAE class. Defines basic API for Neural GraphAE objects.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        build_inference_network(z): Defines encoder network for the graph auto-encoder.
        build_generative_network(z): Defines decoder network for the graph auto-encoder.
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.batch_size = 0
        self.v_sender = None
        self.v_receiver = None

    def build_generative_network(self, z=None):
        raise NotImplementedError

    # Return z_mean and z_log_sigma_sq in all models.
    def build_inference_network(self, x=None):
        raise NotImplementedError

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

    def loss(self):
        """ Wrapper for _loss() """
        return self._loss()


class GCN(GraphAE):
    """ Neural Graph Auto-encoder with GCN encoder and Inner Product decoder. """

    def __init__(self, layers_config, num_features, adj, latent_dim, placeholders, pos_weight, **kwargs):
        super().__init__(**kwargs)
        self.layers_config = layers_config
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.dropout = placeholders['dropout']
        self.pos_weight = pos_weight  # tunable hyper-parameter to re-weight positive links.
        self.batch_size = adj.shape[0]  # Gradient descent over all users.
        self.loss_norm = (self.batch_size * self.batch_size) / \
                         float(self.pos_weight * adj.sum() + (self.batch_size * self.batch_size) - adj.sum())
        a_gcn_tuple = normalize_graph_gcn(adj.astype(np.float32))
        self.A_gcn = tf.SparseTensor(a_gcn_tuple[0].astype(np.float32),
                                     a_gcn_tuple[1].astype(np.float32),
                                     a_gcn_tuple[2])
        a_tuple = sparse_to_tuple(adj)
        self.A = tf.SparseTensor(a_tuple[0].astype(np.float32), a_tuple[1].astype(np.float32), a_tuple[2])
        self.node_indices = tf.range(0, self.batch_size, 1)  # [0,..., N]
        self.v_sender = tf.nn.embedding_lookup(placeholders['v_sender_all'], self.node_indices)
        self.v_receiver = tf.nn.embedding_lookup(placeholders['v_receiver_all'], self.node_indices)

    def _create_inference_network_weights(self):
        """ Create GCN layer parameters for encoder network. """
        self.inference_layers = []
        input_dim = self.num_features
        for i, val in enumerate(self.layers_config):
            self.inference_layers.append(GraphConvolution(input_dim=input_dim,
                                                          output_dim=val,
                                                          adj=self.A_gcn,
                                                          act=tf.nn.relu,
                                                          dropout_rate=self.dropout,
                                                          name="inference_layer_" + str(i)))
            input_dim = val

        # Last layer includes both mu and sigma.
        self.inference_mean_layer = GraphConvolution(input_dim=input_dim,
                                                     output_dim=self.latent_dim,
                                                     adj=self.A_gcn,
                                                     act=tf.nn.relu,
                                                     dropout_rate=self.dropout,
                                                     name="inference_layer_mean")

        self.inference_log_sigma_layer = GraphConvolution(input_dim=input_dim,
                                                          output_dim=self.latent_dim,
                                                          adj=self.A_gcn,
                                                          dropout_rate=self.dropout,
                                                          name="inference_layer_log_sigma",
                                                          act=tf.nn.softplus)

    def build_inference_network(self, x):
        self._build_inference_network(x)

    def build_generative_network(self, z):
        self._build_generative_network(z)

    def _build_inference_network(self, x):
        """ Encoder network: Return z_mean and z_log_sigma_sq after encoding via GCN layers. """
        self._create_inference_network_weights()
        activations = [x]
        for layer in self.inference_layers:
            hidden = layer(activations[-1])
            activations.append(hidden)
        self.z_mean = self.inference_mean_layer(activations[-1])
        self.z_log_sigma_sq = self.inference_log_sigma_layer(activations[-1])
        return self.z_mean, self.z_log_sigma_sq

    def _build_generative_network(self, z):
        """ Decoder network: Returns reconstructed x after decoding via Inner Products. """
        self.x_reconstructed = InnerProductDecoder(act=lambda x: x)(z)
        return self.x_reconstructed

    def _loss(self):
        """ Cross entropy and MSE loss functions for neural graph auto-encoder, with L_2 regularization, """
        self.likelihood_loss = 0.0
        if FLAGS.vae_loss_function == 'cross_entropy':
            predictions = tf.reshape(self.x_reconstructed, [-1])
            targets = tf.reshape(tf.sparse.to_dense(self.A, validate_indices=False), [-1])
            self.likelihood_loss = -self.loss_norm * tf.reduce_mean(
                self.pos_weight * targets * tf.math.log(tf.maximum(predictions, 1e-10)) +
                (1 - targets) * tf.math.log(tf.maximum(1 - predictions, 1e-10)))

        else:
            # MSE loss (with positive re-weighting)
            target_matrix = tf.sparse.to_dense(self.A, validate_indices=False)
            self.B = tf.ones(target_matrix.get_shape(), tf.float32)
            self.B = self.B + tf.cast(target_matrix > 1e-10, self.B.dtype) * (self.pos_weight - 1.0)
            self.likelihood_loss = self.loss_norm * tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(self.B,
                                tf.square(target_matrix - self.x_reconstructed)), 1))

        self.reg_loss = 0.0  # L_2 regularization loss.
        for layer in self.inference_layers:
            for var in layer.vars.values():
                self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for var in self.inference_mean_layer.vars.values():
            self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for var in self.inference_log_sigma_layer.vars.values():
            self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        return self.likelihood_loss, self.reg_loss


class MLP(GraphAE):
    """ Neural Graph Autoencoder with MLP encoder and MLP decoder. """

    def __init__(self, layers_config, num_features, adj, latent_dim, placeholders, pos_weight, **kwargs):
        super().__init__(**kwargs)
        self.layers_config = layers_config
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.dropout = placeholders['dropout']
        self.pos_weight = pos_weight
        self.A = adj.astype(np.float32)
        self._create_batch_queues()
        self.v_sender = tf.nn.embedding_lookup(placeholders['v_sender_all'], self.node_indices)
        self.v_receiver = tf.nn.embedding_lookup(placeholders['v_receiver_all'], self.node_indices)

    def _create_inference_network_weights(self):
        """ Create MLP layer parameters for encoder network. """
        self.inference_layers = []
        input_dim = self.num_features
        for i, val in enumerate(self.layers_config):
            self.inference_layers.append(CustomDense(input_dim=input_dim,
                                                     output_dim=val,
                                                     dropout_rate=self.dropout,
                                                     name="inference_layer_" + str(i)))
            input_dim = val

        # Last layer includes both mu and sigma.
        self.inference_mean_layer = CustomDense(input_dim=input_dim,
                                                output_dim=self.latent_dim,
                                                dropout_rate=self.dropout,
                                                name="inference_layer_mean")

        self.inference_log_sigma_layer = CustomDense(input_dim=input_dim,
                                                     output_dim=self.latent_dim,
                                                     dropout_rate=self.dropout,
                                                     name="inference_layer_log_sigma",
                                                     act=tf.nn.tanh)

    def _create_generative_network_weights(self):
        """ Create MLP layer parameters for decoder network. """
        self.generative_layers = []
        input_dim = self.latent_dim
        for i, val in enumerate(reversed(self.layers_config)):
            self.generative_layers.append(CustomDense(input_dim=input_dim,
                                                      output_dim=val,
                                                      dropout_rate=self.dropout,
                                                      name="generative_layer_" + str(i)))
            input_dim = val

        # Last layer includes just mu.
        self.generative_mean_layer = CustomDense(input_dim=input_dim,
                                                 output_dim=self.num_features,
                                                 dropout_rate=self.dropout,
                                                 name="generative_layer_mean",
                                                 act=tf.nn.sigmoid)

    def build_inference_network(self, x):
        self._build_inference_network()

    def build_generative_network(self, z):
        self._build_generative_network(z)

    def _build_inference_network(self):
        """ Encoder network: Return z_mean and z_log_sigma_sq after encoding via MLP layers. """
        self._create_inference_network_weights()
        activations = [self.node_features]  # This assumes as input [batch_size, num_features]
        for layer in self.inference_layers:
            hidden = layer(activations[-1])
            activations.append(hidden)
        self.z_mean = self.inference_mean_layer(activations[-1])
        self.z_log_sigma_sq = self.inference_log_sigma_layer(activations[-1])
        return self.z_mean, self.z_log_sigma_sq

    def _build_generative_network(self, z):
        """ Decoder network: Returns reconstructed x after decoding via MLP layers. """
        self._create_generative_network_weights()
        activations = [z]
        for layer in self.generative_layers:
            hidden = layer(activations[-1])
            activations.append(hidden)
        self.x_reconstructed = self.generative_mean_layer(activations[-1])
        return self.x_reconstructed

    def _loss(self):
        """ Cross entropy and MSE loss functions for neural graph auto-encoder, with L_2 regularization. """
        self.likelihood_loss = 0.0
        if FLAGS.vae_loss_function == 'cross_entropy':
            self.likelihood_loss = -self.loss_norm * tf.reduce_mean(
                tf.reduce_sum(
                    self.pos_weight * self.node_features * tf.math.log(tf.maximum(self.x_reconstructed, 1e-10))
                    + (1 - self.node_features) * tf.math.log(tf.maximum(1 - self.x_reconstructed, 1e-10)), 1))
        else:
            # Note: do not use sigmoid when rmse is used.
            self.B = tf.ones(self.node_features.get_shape(), tf.float32)
            self.B = self.B + tf.cast(self.node_features > 1e-10, self.B.dtype) * (
                    self.pos_weight - 1.0)  # all pos become pos_weight.
            self.likelihood_loss = self.loss_norm * tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(self.B,
                                tf.square(self.node_features - self.x_reconstructed)),
                    1))

        self.reg_loss = 0.0  # L_2 regularization loss.
        for layer in self.inference_layers:
            for var in layer.vars.values():
                self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for layer in self.generative_layers:
            for var in layer.vars.values():
                self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for var in self.inference_mean_layer.vars.values():
            self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for var in self.inference_log_sigma_layer.vars.values():
            self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        for var in self.generative_mean_layer.vars.values():
            self.reg_loss += FLAGS.vae_weight_decay * tf.nn.l2_loss(var)
        return self.likelihood_loss, self.reg_loss

    def get_sample(self, indices):
        """ Python function to sample rows from adjacency matrix (converted to tensor using tf.py_func). """
        return np.squeeze(self.A[indices, :].toarray())

    def _create_batch_queues(self):
        """ Create batches of rows of laplacian matrix -- can be used for any encoder formulation that involves rows
        of a matrix. """
        num_threads = 10
        node_list = list(range(0, self.A.shape[0]))
        [indices] = tf.train.slice_input_producer([tf.constant(node_list)], shuffle=False,
                                                  capacity=FLAGS.vae_batch_size)
        single_sample = tf.py_func(self.get_sample, [indices], tf.float32)
        min_after_dequeue = 3 * FLAGS.vae_batch_size  # Buffer size maintained in queue after dequeue.
        capacity = min_after_dequeue + (num_threads + 1) * FLAGS.vae_batch_size
        # Initialize TF queue with random shuffling.
        queue = tf.queue.RandomShuffleQueue(capacity=capacity,
                                            dtypes=[tf.float32, tf.int32],
                                            shapes=[self.num_features, indices.get_shape()],
                                            min_after_dequeue=min_after_dequeue)
        enqueue_ops = [queue.enqueue([single_sample, indices])] * num_threads
        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)
        data_batch = queue.dequeue_many(FLAGS.vae_batch_size)  # this has node_features and node_indices
        self.node_features, self.node_indices = data_batch
        self.batch_size = tf.shape(self.node_features)[0]
        mat_sum = tf.cast(tf.reduce_sum(self.num_features), tf.float32)
        prod = (tf.cast(self.batch_size, tf.float32) * tf.cast(self.num_features, tf.float32))
        self.loss_norm = prod / (self.pos_weight * mat_sum + prod - mat_sum)
