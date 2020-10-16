from __future__ import division, print_function

import operator
import time
from pprint import pprint

from models.infvae_models import InfVAESocial, InfVAECascades
from utils.preprocess import *
from utils.flags import *

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# Set logs directory parameters.
LOG_DIR = "log/"
OUTPUT_DATA_DIR = "log/output/"
ensure_dir(LOG_DIR)
ensure_dir(OUTPUT_DATA_DIR)
datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()
log_file = LOG_DIR + '%s_%s_%s_%s.log' % (FLAGS.dataset.split(
 "/")[0], str(today.year), str(today.month), str(today.day))

#--

def predict(session, model, feed):
    """ Helper function to compute model predictions. """
    recall_scores, map_scores, n_samples, top_k, target = \
        session.run([model.recall_scores, model.map_scores, model.relevance_scores,
                     model.top_k_filter, model.targets], feed_dict=feed)
    return recall_scores, map_scores, n_samples.shape[0], top_k, target


with ExpLogger("Inf-VAE", log_file=log_file, data_dir=OUTPUT_DATA_DIR) as logger:
    # log training parameters
    try:
        logger.log(FLAGS.flag_values_dict())
    except:
        logger.log(FLAGS.__flags.items())

    ''' Load data: the datasets are expected to be pre-processed in an appropriate format. The assumption is that the 
    users appearing in cascades must also appear in the graph, while the converse may not be true. 
    Thus, the user indices are created based on the graph. '''
    A = load_graph(FLAGS.dataset)
    if FLAGS.use_feats:
        X = load_feats(FLAGS.dataset)
    else:
        X = np.eye(A.shape[0])

    num_nodes = A.shape[0]
    layers_config = list(map(int, FLAGS.vae_layer_config.split(",")))

    if num_nodes % FLAGS.vae_batch_size == 0:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size
    else:
        num_batches_vae = num_nodes // FLAGS.vae_batch_size + 1

    if FLAGS.graph_AE == 'GCN':
        num_batches_vae = 1

    train_cascades, train_times = load_cascades(FLAGS.dataset, mode='train')
    val_cascades, val_times = load_cascades(FLAGS.dataset, mode='val')
    test_cascades, test_times = load_cascades(FLAGS.dataset, mode='test')

    # Truncating input data based on max_seq_length.
    train_examples, train_examples_times = get_data_set(train_cascades, train_times,
                                                        max_len=FLAGS.max_seq_length,
                                                        mode='train')
    val_examples, val_examples_times = get_data_set(val_cascades, val_times,
                                                    max_len=FLAGS.max_seq_length,
                                                    mode='val')
    test_examples, test_examples_times = get_data_set(test_cascades, test_times,
                                                      max_len=FLAGS.max_seq_length,
                                                      test_min_percent=FLAGS.test_min_percent,
                                                      test_max_percent=FLAGS.test_max_percent,
                                                      mode='test')

    print("# nodes in graph", num_nodes)
    print("# train cascades", len(train_cascades))

    print("Init models")
    VGAE = InfVAESocial(X.shape[1], A, layers_config, mode='train', feats=X)
    CoAtt = InfVAECascades(num_nodes + 1, train_examples, train_examples_times,
                           val_examples, val_examples_times,
                           test_examples, test_examples_times,
                           logging=True, mode='feed')

    # Initialize session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Init variables
    print("Run global var initializer")
    sess.run(tf.global_variables_initializer())

    print("Starting queue runners")
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    z_vae_embeds = np.zeros([num_nodes + 1, FLAGS.latent_dim])
    logger.log("======VAE Pre-train=======")
    # Step 0: Pre-training using simple VAE on social network.
    for epoch in range(FLAGS.pretrain_epochs):
        losses = []
        for b in range(0, num_batches_vae):
            # Training step
            input_feed = VGAE.construct_feed_dict(
                v_sender_all=z_vae_embeds,
                v_receiver_all=z_vae_embeds,
                pre_train=True,
                dropout=FLAGS.vae_dropout_rate)
            _, vae_embeds, indices, train_loss = sess.run([
                VGAE.opt_op, VGAE.z_mean, VGAE.node_indices, VGAE.vae_pre_train_loss
            ], input_feed)
            z_vae_embeds[indices] = vae_embeds
            losses.append(train_loss)
        epoch_loss = np.mean(losses)
        logger.log("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))
    logger.log("Pre-training completed")

    # Initial run to get embeddings.
    logger.log("Initial run to get embeddings")
    for b in range(0, num_batches_vae):
        t = time.time()
        indices, z_val = sess.run([VGAE.node_indices, VGAE.z_mean])
        z_vae_embeds[indices] = z_val
        s = time.time()

    val_loss_all = []
    sender_embeds = np.copy(z_vae_embeds)
    receiver_embeds = np.copy(z_vae_embeds)
    for epoch in range(FLAGS.epochs):
        # Train
        # Step 1: VAE on Social Network.
        losses = []
        input_feed = VGAE.construct_feed_dict(
            v_sender_all=sender_embeds,
            v_receiver_all=receiver_embeds,
            dropout=FLAGS.vae_dropout_rate)
        for b in range(0, num_batches_vae):
            # Training step
            _, vae_embeds, indices, train_loss = sess.run([VGAE.opt_op, VGAE.z_mean, VGAE.node_indices,
                                                           VGAE.social_loss], input_feed)
            z_vae_embeds[indices] = vae_embeds
            losses.append(train_loss)

        epoch_loss = np.mean(losses)
        logger.log("Mean VAE loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))

        # Step 2: Diffusion Cascades
        losses = []
        input_feed = CoAtt.construct_feed_dict(z_vae_embeddings=z_vae_embeds)

        for b in range(0, CoAtt.num_train_batches):
            _, train_loss = sess.run([CoAtt.opt_op, CoAtt.diffusion_loss], input_feed)
            losses.append(train_loss)
        # re-assign based on updated sender, receiver embeddings.
        sender_embeds = sess.run(CoAtt.sender_embeddings)
        receiver_embeds = sess.run(CoAtt.receiver_embeddings)
        epoch_loss = np.mean(losses)
        logger.log("Mean Attention loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))

        # Testing
        if epoch % FLAGS.test_freq == 0:
            input_feed = VGAE.construct_feed_dict(v_sender_all=sender_embeds,
                                                  v_receiver_all=receiver_embeds, dropout=0.)
            for _ in range(0, num_batches_vae):
                vae_embeds, indices = sess.run([VGAE.z_mean, VGAE.node_indices], input_feed)
                z_vae_embeds[indices] = vae_embeds
            input_feed = CoAtt.construct_feed_dict(z_vae_embeddings=z_vae_embeds, is_test=True)

            total_samples = 0
            num_eval_k = len(CoAtt.k_list)
            avg_map_scores, avg_recall_scores = [0.] * num_eval_k, [0.] * num_eval_k

            all_outputs, all_targets = [], []
            for b in range(0, CoAtt.num_test_batches):
                recalls, maps, num_samples, decoder_outputs, decoder_targets = predict(
                    sess, CoAtt, input_feed)
                all_outputs.append(decoder_outputs)
                all_targets.append(decoder_targets)
                avg_map_scores = list(
                    map(operator.add, map(operator.mul, maps,
                                          [num_samples] * num_eval_k), avg_map_scores))
                avg_recall_scores = list(map(operator.add, map(operator.mul, recalls,
                                                               [num_samples] * num_eval_k), avg_recall_scores))
                total_samples += num_samples
            all_outputs = np.vstack(all_outputs)
            all_targets = np.vstack(all_targets)
            avg_map_scores = list(map(operator.truediv, avg_map_scores, [total_samples] * num_eval_k))
            avg_recall_scores = list(map(operator.truediv, avg_recall_scores, [total_samples] * num_eval_k))

            metrics = dict()
            for k in range(0, num_eval_k):
                K = CoAtt.k_list[k]
                metrics["MAP@%d" % K] = avg_map_scores[k]
                metrics["Recall@%d" % K] = avg_recall_scores[k]

            logger.update_record(avg_map_scores[0], (all_outputs, all_targets, metrics))

        # Validation
        if epoch % FLAGS.val_freq == 0:
            input_feed = VGAE.construct_feed_dict(
                v_sender_all=sender_embeds, v_receiver_all=receiver_embeds, dropout=0.)
            for b in range(0, num_batches_vae):
                vae_embeds, indices = sess.run([VGAE.z_mean, VGAE.node_indices], input_feed)
                z_vae_embeds[indices] = vae_embeds
            losses = []
            num_eval_k = len(CoAtt.k_list)
            input_feed = CoAtt.construct_feed_dict(z_vae_embeddings=z_vae_embeds, is_val=True)
            for b in range(0, CoAtt.num_val_batches):
                val_loss = sess.run([CoAtt.diffusion_loss], input_feed)
                losses.append(val_loss)
            epoch_loss = np.mean(losses)
            val_loss_all.append(epoch_loss)
            logger.log("Validation Attention loss at epoch: %04d %.5f" % (epoch + 1, epoch_loss))

            # early stopping
            if len(val_loss_all) >= FLAGS.early_stopping and val_loss_all[-1] > np.mean(
                    val_loss_all[-(FLAGS.early_stopping + 1):-1]):
                logger.log("Early stopping at epoch: %04d" % (epoch + 1))
                break

    # print evaluation metrics
    outputs, targets, metrics = logger.best_data
    print("Evaluation metrics on test set:")
    pprint(metrics)

    # stop queue runners
    coord.request_stop()
    coord.join(threads)
