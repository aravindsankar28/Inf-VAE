from __future__ import print_function

import logging
import os
import pickle
from datetime import datetime

import networkx as nx
import numpy as np
import scipy.sparse as sp

import pandas as pd
import tensorflow as tf

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class ExpLogger:
    """ Experiment logger. """
    def __init__(self, name, cmd_print=True, log_file=None, spreadsheet=None, data_dir=None):
        self.datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.name = name + "_" + self.datetime_str
        self.cmd_print = cmd_print
        log_level = logging.INFO
        logging.basicConfig(filename=log_file, level=log_level,
                            format='%(asctime)s - %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S')
        self.file_logger = logging.getLogger()
        self.spreadsheet = spreadsheet
        self.data_dir = data_dir
        if self.spreadsheet is not None:
            dirname = os.path.dirname(self.spreadsheet)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if os.path.isfile(self.spreadsheet):
                try:
                    self.df = pd.read_csv(spreadsheet)
                except:
                    self.df = pd.DataFrame()
            else:
                self.df = pd.DataFrame()
        if self.data_dir is not None:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
        self.best_metric = float("-inf")
        self.best_data = None

    def __enter__(self):
        self.log("Logger Started, name: " + self.name)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.spreadsheet is not None:
            self.df.to_csv(self.spreadsheet, index=False)

    def log(self, content):
        if not isinstance(content, str):
            content = str(content)
        if self.cmd_print:
            print(content)
        if self.file_logger is not None:
            self.file_logger.info(content)

    def debug(self, content):
        if not isinstance(content, str):
            content = str(content)
        if self.cmd_print:
            print("[DEBUG]::: " + content + ":::[DEBUG]")
        if self.file_logger is not None:
            self.file_logger.debug(str)

    def spreadsheet_write(self, val_dict):
        if self.spreadsheet is not None:
            if "name" not in val_dict:
                val_dict["name"] = self.name
            self.df = self.df.append(val_dict, ignore_index=True)

    def save_data(self, data, name):
        name = name + "_" + self.datetime_str
        if isinstance(data, np.ndarray):
            np.savez_compressed(os.path.join(self.data_dir, name + ".npz"), data=data)
        else:
            with open(os.path.join(self.data_dir, name + ".pkl"), "wb") as f:
                pickle.dump(data, f)

    def update_record(self, metric, data):
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_data = data


def sparse_to_tuple(sparse_mx):
    """ Convert sparse matrix to tuple representation. """

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        """ Convert a list of sparse matrices to tuple representation. """
        coords = []
        values = []
        shape = [len(matrices)]
        for idx in range(0, len(matrices)):
            mx = matrices[idx]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * idx]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def normalize_graph_gcn(adj):
    """ Normalize adjacency matrix following GCN. """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    return sparse_to_tuple(adj_normalized)


def sample_mask(idx, l):
    """ Create mask. """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


extra_tokens = ['_GO', 'EOS']


def get_data_set(cascades, timestamps, max_len=None, test_min_percent=0.1, test_max_percent=0.5, mode='test'):
    """ Create train/val/test examples from input cascade sequences. Cascade sequences are truncated based on max_len.
    Test examples are sampled with seed set percentage between 10% and 50%. Train/val sets include examples of all
    possible seed sizes. """
    dataset, dataset_times = [], []
    eval_set, eval_set_times = [], []
    for cascade in cascades:
        if max_len is None or len(cascade) < max_len:
            dataset.append(cascade)
        else:
            dataset.append(cascade[0:max_len])  # truncate

    for ts_list in timestamps:
        if max_len is None or len(ts_list) < max_len:
            dataset_times.append(ts_list)
        else:
            dataset_times.append(ts_list[0:max_len])  # truncate

    for cascade, ts_list in zip(dataset, dataset_times):
        assert len(cascade) == len(ts_list)
        for j in range(1, len(cascade)):
            seed_set = cascade[0:j]
            seed_set_times = ts_list[0:j]
            remain = cascade[j:]
            remain_times = ts_list[j:]
            seed_set_percent = len(seed_set) / (len(seed_set) + len(remain))
            if mode == 'train' or mode == 'val':
                eval_set.append((seed_set, remain))
                eval_set_times.append((seed_set_times, remain_times))
            if mode == 'test' and (test_min_percent < seed_set_percent < test_max_percent):
                eval_set.append((seed_set, remain))
                eval_set_times.append((seed_set_times, remain_times))
    print("# {} examples {}".format(mode, len(eval_set)))
    return eval_set, eval_set_times


def load_graph(dataset_str):
    """ Load social network as a sparse adjacency matrix. """
    print("Loading graph", dataset_str)
    g = nx.Graph()
    n_nodes, n_edges = 0, 0
    with open("data/{}/{}".format(dataset_str, "graph.txt"), 'rb') as f:
        nu = 0
        for line in f:
            nu += 1
            if nu == 1:
                # assuming first line contains number of nodes, edges.
                n_nodes, n_edges = [int(x) for x in line.strip().split()]
                for i in range(n_nodes):
                    g.add_node(i)
                continue
            s, t = [int(x) for x in line.strip().split()]
            g.add_edge(s, t)
    adj = nx.adjacency_matrix(g)
    print("# nodes", n_nodes, "# edges", n_edges, adj.shape)
    global start_token, end_token
    start_token = adj.shape[0] + extra_tokens.index('_GO')  # start_token = 0
    end_token = adj.shape[0] + extra_tokens.index('EOS')  # end_token = 1
    return adj


def load_feats(dataset_str):
    """ Load user attributes in social network. """
    x = np.load("data/{}/{}".format(dataset_str, "feats.npz"))
    return x['arr_0']


def load_cascades(dataset_str, mode='train'):
    """ Load cascade data, return cascade (user sequence, timestamp sequence). """
    print("Loading cascade", dataset_str, "mode", mode)
    cascades = []
    global avg_diff
    avg_diff = 0.0
    time_stamps = []
    path = mode + str(".txt")
    with open("data/{}/{}".format(dataset_str, path), 'rb') as f:
        for line in f:
            if len(line) < 1:
                continue
            line = list(map(float, line.split()))
            start, rest = int(line[0]), line[1:]
            cascade = [start]
            cascade.extend(list(map(int, rest[::2])))
            time_stamp = [0]  # start time = 0
            time_stamp.extend(rest[1::2])
            cascades.append(cascade)
            time_stamps.append(time_stamp)
    return cascades, time_stamps


def prepare_batch_sequences(input_sequences, target_sequences, batch_size):
    """ Split cascade sequences into batches based on batch_size. """
    # Split based on batch_size
    assert (len(input_sequences) == len(target_sequences))
    num_batch = len(input_sequences) // batch_size
    if len(input_sequences) % batch_size != 0:
        num_batch += 1
    batches_x = []
    batches_y = []
    n = len(input_sequences)
    for i in range(0, num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batches_x.append(input_sequences[start:end])
        batches_y.append(target_sequences[start:end])
    return batches_x, batches_y


def prepare_batch_graph(adj, batch_size):
    """ Split users into batches based on batch_size. """
    n = adj.shape[0]
    num_batch = n // batch_size + 1
    random_ordering = np.random.permutation(n)
    batches = []
    batches_indices = []
    for i in range(0, num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, n)
        batch_indices = random_ordering[start:end]
        batch = adj[batch_indices, :]
        batches.append(batch.toarray())
        batches_indices.append(batch_indices)
    return batches, batches_indices


def prepare_sequences(examples, examples_times, max_len=None, cascade_batch_size=1, mode='train'):
    """ Prepare sequences by padding and adding dummy evaluation sequences. """
    seqs_x = list(map(lambda seq_t: (seq_t[0][(-1) * max_len:], seq_t[1]), examples))
    times_x = list(map(lambda seq_t: (seq_t[0][(-1) * max_len:], seq_t[1]), examples_times))
    # add padding.
    lengths_x = [len(s[0]) for s in seqs_x]
    lengths_y = [len(s[1]) for s in seqs_x]

    if len(seqs_x) % cascade_batch_size != 0 and (mode == 'test' or mode == 'val'):
        # Dummy sequences for evaluation: this is required to ensure that each batch is full-sized -- else the
        # data may not be split perfectly while evaluation.
        x_batch_size = (1 + len(seqs_x) // cascade_batch_size) * cascade_batch_size
        lengths_x.extend([1] * (x_batch_size - len(seqs_x)))
        lengths_y.extend([1] * (x_batch_size - len(seqs_x)))

    x_lengths = np.array(lengths_x).astype('int32')
    max_len_x = max_len
    # mask input with start token (n_nodes + 1) to work with embedding_lookup
    x = np.ones((len(lengths_x), max_len_x)).astype('int32') * start_token
    # mask target with -1 so that tf.one_hot will return a zero vector for padded nodes
    y = np.ones((len(lengths_y), max_len_x)).astype('int32') * -1
    # activation times are set to vector of ones.
    x_times = np.ones((len(lengths_x), max_len_x)).astype('int32') * -1
    y_times = np.ones((len(lengths_y), max_len_x)).astype('int32') * -1
    mask = np.ones_like(x)

    # Assign final set of sequences.
    for idx, (s_x, t) in enumerate(seqs_x):
        end_x = lengths_x[idx]
        end_y = lengths_y[idx]
        x[idx, :end_x] = s_x
        y[idx, :end_y] = t
        mask[idx, end_x:] = 0

    for idx, (s_x, t) in enumerate(times_x):
        end_x = lengths_x[idx]
        end_y = lengths_y[idx]
        x_times[idx, :end_x] = s_x
        y_times[idx, :end_y] = t

    return x, x_lengths, y, mask, x_times, y_times


def ensure_dir(d):
    """ Helper function to create directory if it does not exist. """
    if not os.path.isdir(d):
        os.makedirs(d)
