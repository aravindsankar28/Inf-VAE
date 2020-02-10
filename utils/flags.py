import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# General params.
flags.DEFINE_string('dataset', 'christianity', 'Dataset string.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('pretrain_epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')
flags.DEFINE_string('cuda_device', '0', 'GPU in use')

flags.DEFINE_integer('test_freq', 1, 'Testing frequency')
flags.DEFINE_integer('val_freq', 1, 'Validation frequency')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_integer('batch_queue_threads', 8, 'Threads used for tf batch queue')

flags.DEFINE_string('graph_AE', 'GCN', 'Graph AutoEncoder Options (MLP, GCN)')
flags.DEFINE_boolean('use_feats', False, 'Use features in GCN or not')

# Model Hyper-parameters.
flags.DEFINE_float('lambda_s', 1.0, 'Lambda_s')
flags.DEFINE_float('lambda_r', 0.01, 'Lambda_r')
flags.DEFINE_integer('latent_dim', 64, 'Latent embedding dimension')
flags.DEFINE_float('pos_weight', 1, 'Pos weight for cross entropy. -1 decides automatically')

# Evaluation parameters.
flags.DEFINE_integer('max_seq_length', 100, 'Maximum sequence length')
flags.DEFINE_float('test_min_percent', 0.1, 'Minimum seed set percentage for testing.')
flags.DEFINE_float('test_max_percent', 0.5, 'Maximum seed set percentage for testing.')


# Co-Attention model parameters.
flags.DEFINE_float('lambda_a', 0.1, 'Lambda_a for attention weights')
flags.DEFINE_float('cascade_lr', 0.01, 'Initial learning rate for cascade model.')
flags.DEFINE_integer('cascade_batch_size', 64, 'Batch size attention')

# VAE model parameters.
flags.DEFINE_string('vae_layer_config', '256,128,64', 'VAE NN layer config: comma separated number of units')
flags.DEFINE_float('vae_dropout_rate', 0.2, 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('vae_loss_function', 'cross_entropy', 'Loss function for VGAE: (rmse,cross_entropy)')
flags.DEFINE_integer('vae_batch_size', 64, 'Batch size VAE')
flags.DEFINE_float('vae_lr', 0.01, 'Initial learning rate for vae model.')
flags.DEFINE_float('vae_weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
