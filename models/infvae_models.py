from __future__ import division, print_function

import math
from eval.eval_metrics import *
from models.graph_ae import GCN, MLP
from models.models import Model
from utils.preprocess import *

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class InfVAECascades(Model):
    """ Diffusion Cascades component of Inf-VAE. """

    def __init__(self, num_nodes, train_examples, train_examples_times, val_examples, val_examples_times,
                 test_examples, test_examples_times, mode='feed', **kwargs):
        super(InfVAECascades, self).__init__(**kwargs)
        self.k_list = [10, 50, 100]  # size of rank list for evaluation.
        # Prepare train, test, and val examples -- use max_seq_length --
        train_examples, train_lengths, train_targets, train_masks, train_examples_times, train_targets_times = \
            prepare_sequences(train_examples, train_examples_times, max_len=FLAGS.max_seq_length, mode='train')

        val_examples, val_lengths, val_targets, val_masks, val_examples_times, val_targets_times = \
            prepare_sequences(val_examples, val_examples_times, max_len=FLAGS.max_seq_length, mode='val')

        test_examples, test_lengths, test_targets, test_masks, test_examples_times, test_targets_times = \
            prepare_sequences(test_examples, test_examples_times, max_len=FLAGS.max_seq_length, mode='test')

        # Compute number of batches.
        num_train, num_val, num_test = len(train_examples), len(val_examples), len(test_examples)

        self.num_train_batches = num_train // FLAGS.cascade_batch_size
        if num_train % FLAGS.cascade_batch_size != 0:
            self.num_train_batches += 1

        self.num_val_batches = num_val // FLAGS.cascade_batch_size
        if num_val % FLAGS.cascade_batch_size != 0:
            self.num_val_batches += 1

        self.num_test_batches = num_test // FLAGS.cascade_batch_size
        if num_test % FLAGS.cascade_batch_size != 0:
            self.num_test_batches += 1

        self.lambda_s, self.lambda_r, self.lambda_a = FLAGS.lambda_s, FLAGS.lambda_r, FLAGS.lambda_a
        self.embedding_size = FLAGS.latent_dim
        self.mode = mode

        self.inputs_train, self.targets_train = train_examples, train_targets
        self.inputs_length_train, self.masks_train = train_lengths, train_masks
        self.inputs_train_times, self.targets_train_times = train_examples_times, train_targets_times

        self.inputs_val, self.targets_val = val_examples, val_targets
        self.inputs_length_val, self.masks_val = val_lengths, val_masks
        self.inputs_val_times, self.targets_val_times = val_examples_times, val_targets_times

        self.inputs_test, self.targets_test = test_examples, test_targets
        self.inputs_length_test, self.masks_test = test_lengths, test_masks
        self.inputs_test_times, self.targets_test_times = test_examples_times, test_targets_times

        self.num_nodes = num_nodes
        self.global_step, self.is_val, self.is_test, self.z_vae_embeddings = self._init_placeholders()

        self.inputs, self.inputs_length, self.targets, self.masks, self.inputs_times, self.targets_times = \
            self.create_batch_queues()

        self.batch_size = tf.shape(self.inputs)[0]
        self.build()

    def construct_feed_dict(self, z_vae_embeddings=None, is_val=False, is_test=False):
        """ Construct minimal feed dict with val/test flags and fixed social embeddings. """
        input_feed = {self.is_val.name: is_val, self.is_test.name: is_test}
        if z_vae_embeddings is not None:
            input_feed[self.z_vae_embeddings.name] = z_vae_embeddings
        return input_feed

    def create_batch_queues(self):
        """ Create batch queues in TF to efficiently feed in input and target sequences with (sequence,
        target) pairs, along with masks. """
        num_threads = FLAGS.batch_queue_threads

        # Define train/val/test data batches by reading from input sequences using slice_input_producer.
        [input_train, length_train, target_train, masks_train, input_train_times, target_train_times] = \
            tf.train.slice_input_producer([self.inputs_train, self.inputs_length_train,
                                           self.targets_train, self.masks_train,
                                           self.inputs_train_times, self.targets_train_times],
                                          shuffle=True, capacity=FLAGS.cascade_batch_size)

        [input_val, length_val, target_val, masks_val, input_val_times, target_val_times] = \
            tf.train.slice_input_producer([self.inputs_val, self.inputs_length_val,
                                           self.targets_val, self.masks_val,
                                           self.inputs_val_times, self.targets_val_times],
                                          shuffle=False, capacity=FLAGS.cascade_batch_size)

        [input_test, length_test, target_test, masks_test, input_test_times, target_test_times] = \
            tf.train.slice_input_producer([self.inputs_test, self.inputs_length_test, self.targets_test,
                                           self.masks_test, self.inputs_test_times, self.targets_test_times],
                                          shuffle=False, capacity=FLAGS.cascade_batch_size)

        min_after_dequeue = FLAGS.cascade_batch_size
        q_size = min_after_dequeue + (num_threads + 1) * FLAGS.cascade_batch_size

        # Initialize train/val/test queues.
        train_queue = tf.queue.RandomShuffleQueue(capacity=q_size, dtypes=[tf.int32] * 6,
                                                  shapes=[input_train.get_shape(), length_train.get_shape(),
                                                          target_train.get_shape(), masks_train.get_shape(),
                                                          input_train_times.get_shape(),
                                                          target_train_times.get_shape()],
                                                  min_after_dequeue=min_after_dequeue)

        val_queue = tf.queue.PaddingFIFOQueue(capacity=FLAGS.cascade_batch_size, dtypes=[tf.int32] * 6,
                                              shapes=[input_val.get_shape(), length_val.get_shape(),
                                                      target_val.get_shape(), masks_val.get_shape(),
                                                      input_val_times.get_shape(), target_val_times.get_shape()])

        test_queue = tf.queue.PaddingFIFOQueue(capacity=FLAGS.cascade_batch_size, dtypes=[tf.int32] * 6,
                                               shapes=[input_test.get_shape(), length_test.get_shape(),
                                                       target_test.get_shape(), masks_test.get_shape(),
                                                       input_test_times.get_shape(), target_test_times.get_shape()])

        # Define train/val/test enqueue operations.
        train_enqueue_ops = [
                                train_queue.enqueue([
                                    input_train, length_train, target_train,
                                    masks_train, input_train_times,
                                    target_train_times
                                ])
                            ] * num_threads

        val_enqueue_ops = [
                              val_queue.enqueue([
                                  input_val, length_val, target_val,
                                  masks_val, input_val_times, target_val_times
                              ])
                          ] * num_threads

        test_enqueue_ops = [
                               test_queue.enqueue([
                                   input_test, length_test, target_test,
                                   masks_test, input_test_times,
                                   target_test_times
                               ])
                           ] * num_threads

        # Define train/val/test queue runners.
        qr_train = tf.train.QueueRunner(train_queue, train_enqueue_ops)
        qr_val = tf.train.QueueRunner(val_queue, val_enqueue_ops)
        qr_test = tf.train.QueueRunner(test_queue, test_enqueue_ops)

        tf.train.add_queue_runner(qr_train)
        tf.train.add_queue_runner(qr_val)
        tf.train.add_queue_runner(qr_test)

        # Feed train/val/test based on `is_val' and `is_test' flag.
        data_batch = tf.case(
            {
                self.is_val:
                    lambda: val_queue.dequeue_many(FLAGS.cascade_batch_size),
                self.is_test:
                    lambda: test_queue.dequeue_many(FLAGS.cascade_batch_size)
            },
            default=lambda: train_queue.dequeue_many(FLAGS.cascade_batch_size), exclusive=True)

        return data_batch

    def _init_placeholders(self):
        """" Initialize minimal set of placeholders train/val/test flags, and fixed social embeddings. """
        global_step = tf.Variable(0, trainable=False, name='global_step')
        is_val = tf.compat.v1.placeholder(tf.bool, name='is_val')
        is_test = tf.compat.v1.placeholder(tf.bool, name='is_test')
        z_vae_embeddings = None
        if self.mode == 'feed':
            # Add z_vae_embeddings input.
            z_vae_embeddings = tf.compat.v1.placeholder(
                dtype=tf.float32,
                shape=(self.num_nodes, FLAGS.latent_dim),
                name='z_vae_embeddings')
        return global_step, is_val, is_test, z_vae_embeddings

    @staticmethod
    def zero_out(tensor, mask_value=-1, scope=None):
        """Zero out mask value from the given tensor.
        Args:
            tensor: `Tensor` of size [batch_size, seq_length] of target indices.
            mask_value: Missing label value.
            scope: `string`, scope of the operation.
        Returns:
            targets: Targets with zerod-out values.
            mask: Mask values.
        """
        with tf.compat.v1.variable_scope(scope, default_name='zero_out'):
            in_vocab_indicator = tf.not_equal(tensor, mask_value)
            tensor *= tf.cast(in_vocab_indicator, tensor.dtype)
            mask = tf.to_float(in_vocab_indicator)
        return tensor, mask

    def _build_co_attention(self):
        """ Co-attentive diffusion cascade generation. """
        with tf.compat.v1.variable_scope('co-attention'):
            sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            # Define all trainable parameters.
            self.temporal_embeddings = tf.compat.v1.get_variable(
                name='temporal_embedding',
                shape=[self.num_nodes, self.embedding_size],
                initializer=initializer)

            self.receiver_embeddings = tf.compat.v1.get_variable(
                name='receiver_embedding',
                shape=[self.num_nodes, self.embedding_size],
                initializer=initializer)

            self.sender_embeddings = tf.compat.v1.get_variable(
                name='sender_embedding',
                shape=[self.num_nodes, self.embedding_size],
                initializer=initializer)

            self.co_attn_wts = tf.compat.v1.get_variable(
                name='co_attention_weights',
                shape=[self.embedding_size, self.embedding_size],
                initializer=initializer)

            self.position_embeddings = tf.compat.v1.get_variable(
                name='position_embeddings',
                shape=[FLAGS.max_seq_length, self.embedding_size],
                initializer=initializer)

            self.sender_embedded = tf.nn.embedding_lookup(
                params=self.sender_embeddings,
                ids=self.inputs)  # (batch_size, seq_len, embed_size)

            self.temporal_embedded = tf.nn.embedding_lookup(
                params=self.temporal_embeddings,
                ids=self.inputs)  # (batch_size, seq_len, embed_size)

            # Mask input sequence.
            self.sender_embedded = self.sender_embedded * tf.expand_dims(tf.cast(self.masks, tf.float32), -1)
            self.temporal_embedded = self.temporal_embedded * tf.expand_dims(tf.cast(self.masks, tf.float32), -1)
            self.temporal_embedded = self.temporal_embedded + self.position_embeddings

            attn_act = tf.nn.tanh(
                tf.reduce_sum(tf.multiply(
                    tf.tensordot(self.sender_embedded, self.co_attn_wts,
                                 axes=[[2], [0]]), self.temporal_embedded), 2))  # (batch_size, seq_len)

            attn_alpha = tf.nn.softmax(attn_act)  # (batch_size, seq_len)
            self.attended_embeddings = self.temporal_embedded * tf.expand_dims(attn_alpha, -1)
            self.attended_embeddings = tf.reduce_sum(self.attended_embeddings, 1)  # (batch_size, embed_size)

            self.outputs = tf.matmul(self.attended_embeddings, tf.transpose(
                self.receiver_embeddings))  # (batch_size, num_users)

            _, self.top_k = tf.nn.top_k(self.outputs, k=200)  # (batch_size, 200)

            # Remove seed users from the predicted rank list.
            self.top_k_filter = tf.py_func(remove_seeds, [self.top_k, self.inputs], tf.int32)

            masks = tf.cast(tf.reshape(
                tf.py_func(get_masks, [self.top_k_filter, self.inputs],
                           tf.int32), [-1]), tf.bool)

            relevance_scores_all = tf.py_func(get_relevance_scores, [self.top_k_filter, self.targets], tf.bool)

            # Number of relevant candidates.
            m = tf.reduce_sum(tf.reduce_max(tf.one_hot(self.targets, self.num_nodes), axis=1), -1)

            self.relevance_scores = tf.cast(tf.boolean_mask(tf.cast(relevance_scores_all,
                                                                    tf.float32), masks), tf.int32)
            # Metric score computation.
            self.recall_scores = [
                tf.py_func(mean_recall_at_k, [self.relevance_scores, k, m],
                           tf.float32) for k in self.k_list
            ]

            self.map_scores = [
                tf.py_func(MAP, [self.relevance_scores, k, m], tf.float32)
                for k in self.k_list
            ]

    def init_optimizer(self):
        """ Initialize Adam optimizer for Co-attentive cascade model. """
        # Gradients and SGD update operation for training the model
        trainable_params = tf.compat.v1.trainable_variables()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.cascade_lr)
        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(self.diffusion_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   FLAGS.max_gradient_norm)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params),
                                                     global_step=self.global_step)

    def _loss(self):
        """ Define cascade losses (cascade and alignment). """
        # tf.one_hot will return zero vector for -1 padding in targets
        labels_k_hot = tf.reduce_max(tf.one_hot(self.targets, self.num_nodes), axis=1)
        if FLAGS.pos_weight < 0:
            # pos_weight is computed to equally balance positives and negatives.
            self.cascade_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(
                    targets=labels_k_hot,
                    logits=self.outputs,
                    pos_weight=(self.num_nodes / FLAGS.max_seq_length
                                ))) + self.lambda_a * tf.nn.l2_loss(self.co_attn_wts)
        else:
            self.cascade_loss = tf.reduce_mean(
                tf.nn.weighted_cross_entropy_with_logits(targets=labels_k_hot,
                                                         logits=self.outputs,
                                                         pos_weight=FLAGS.pos_weight)
            ) + self.lambda_a * tf.nn.l2_loss(self.co_attn_wts)

        # Sender and receiver losses.
        self.sender_loss = 0.5 * self.lambda_s * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(
                    self.sender_embeddings -
                    self.z_vae_embeddings),
                1))

        self.receiver_loss = 0.5 * self.lambda_r * tf.reduce_mean(
            tf.reduce_sum(
                tf.square(self.receiver_embeddings - self.z_vae_embeddings), 1))

        # Overall diffusion cascade loss.
        self.diffusion_loss = self.cascade_loss + self.sender_loss + self.receiver_loss

    def _build(self):
        self._build_co_attention()
        self._loss()
        self.init_optimizer()


class InfVAESocial(Model):
    """ Social Network component of Inf-VAE. """

    def __init__(self, num_features, adj, vae_layers_config, mode, feats,
                 **kwargs):
        super(InfVAESocial, self).__init__(**kwargs)
        self.mode = mode
        self.num_features = num_features
        self.A = adj.astype(np.float32)
        self.feats = feats.astype(np.float32)
        self.layers_config = vae_layers_config
        self.latent_dim = FLAGS.latent_dim
        self.lambda_s = FLAGS.lambda_s
        self.lambda_r = FLAGS.lambda_r
        self.pos_weight = 100  # Tunable parameter.
        self._init_placeholders()
        if FLAGS.graph_AE == 'MLP':
            model_func = MLP
        elif FLAGS.graph_AE == 'GCN':
            model_func = GCN
        else:
            model_func = None
        self.graph_ae = model_func(layers_config=self.layers_config,
                                   num_features=self.num_features,
                                   adj=adj,
                                   latent_dim=self.latent_dim,
                                   placeholders=self.placeholders,
                                   pos_weight=self.pos_weight)
        self.build()
        self.node_indices = self.graph_ae.node_indices

    # Feed in v_s, v_r, pre_train flag, dropout -- sender and receiver latent embeddings.
    def construct_feed_dict(self, v_sender_all=None, v_receiver_all=None, pre_train=False, dropout=0.):
        """ Construct minimal feed_dict given input sender, receiver embeddings and flags. """
        input_feed = {}
        if v_sender_all is not None and v_receiver_all is not None:
            input_feed[self.sender.name] = v_sender_all
            input_feed[self.receiver.name] = v_receiver_all
        input_feed[self.pre_train.name] = pre_train
        input_feed[self.dropout.name] = dropout
        return input_feed

    def _init_placeholders(self):
        """ Initialize minimal set of placeholders with sender, receiver embeddings and flags. """
        self.pre_train = tf.compat.v1.placeholder(tf.bool, name='pre_train')
        self.dropout = tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout')
        self.sender = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None), name='sender')
        self.receiver = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, None), name='receiver')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.placeholders = {'v_sender_all': self.sender, 'v_receiver_all': self.receiver,
                             'pre_train': self.pre_train, 'dropout': self.dropout}

    def _loss(self):
        """ Define VAE losses (KL divergence, alignment, and graph likelihood). """
        # (a) KL divergence loss
        self.kld_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(
            tf.square(self.z_mean) + tf.exp(self.z_log_sigma_sq) -
            self.z_log_sigma_sq - 1, 1))

        # (b) Alignment losses.
        self.sender_loss = 0.5 * self.lambda_s * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v_sender - self.z_mean), 1))

        self.receiver_loss = 0.5 * self.lambda_r * tf.reduce_mean(
            tf.reduce_sum(tf.square(self.v_receiver - self.z_mean), 1))

        # (c) Social network reconstruction or likelihood loss.
        self.likelihood_loss, self.reg_loss = self.graph_ae.loss()
        self.vae_pre_train_loss = self.likelihood_loss + self.kld_loss + self.reg_loss
        self.social_loss = self.kld_loss + self.likelihood_loss + self.sender_loss + \
                           self.receiver_loss + self.reg_loss

    def _build(self):
        self._build_vae()
        self._loss()
        self.init_optimizer()

    def _build_vae(self):
        """ Define encoder, decoder, and stochastic sampling for VAE. """
        self.graph_ae.build_inference_network(self.feats)
        self.z_mean, self.z_log_sigma_sq = self.graph_ae.z_mean, self.graph_ae.z_log_sigma_sq
        # Draw one sample z from Gaussian distribution
        eps = tf.random.normal((self.graph_ae.batch_size, self.latent_dim), 0, 1, dtype=tf.float32)
        # z = mu + sigma * epsilon
        self.z = tf.add(self.z_mean,
                        tf.multiply(tf.sqrt(tf.maximum(tf.exp(self.z_log_sigma_sq), 1e-10)),
                                    eps))

        self.graph_ae.build_generative_network(self.z)
        self.v_sender, self.v_receiver = self.graph_ae.v_sender, self.graph_ae.v_receiver

    def init_optimizer(self):
        """ Initialize Adam optimizer for VAE model. """
        # Gradients and SGD update operation for training the model
        trainable_params = tf.compat.v1.trainable_variables()
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.vae_lr)
        actual_loss = tf.cond(self.pre_train, lambda: self.vae_pre_train_loss, lambda: self.social_loss)
        # Compute gradients of loss w.r.t. all trainable variables
        gradients = tf.gradients(actual_loss, trainable_params)
        # Clip gradients by a given maximum_gradient_norm
        clip_gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)
        # Set the model optimization op.
        self.opt_op = self.optimizer.apply_gradients(zip(clip_gradients, trainable_params),
                                                     global_step=self.global_step)
