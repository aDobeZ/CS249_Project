# -*- coding: utf-8 -*-

import time
import tensorflow as tf

from utils2 import *
from models import DHNE
import random
from Active_select2 import *
from rewards import *
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'MovieLens', 'Dataset string.')  # 'MovieLens', 'Cora', 'DBLP_four_area'
flags.DEFINE_string('model', 'DHNE', 'Model string.')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
dataset_arr = FLAGS.dataset.split('|')
for data_index in range(len(dataset_arr)):
    data_str = dataset_arr[data_index]
    # Load data
    node_objects, features, network_objects, all_y_index, all_y_label, \
    pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(data_str)
    importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
    # Some preprocessing
    # features = preprocess_features(features)
    features = sparse_to_tuple(features)
    # support = [preprocess_adj(new_adj)]
    support = []
    for index in range(len(network_objects)):
        adj = network_objects[index]
        support.append(preprocess_adj(adj))
    num_supports = len(support)

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, class_num)),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Define model training function
    def DHNE_train(y_train1, train_mask1, y_val1, val_mask1, y_test1, test_mask1):
        model_t = time.time()
        # Create model
        model = DHNE(placeholders, input_dim=features[2][1], logging=True)
        # Initialize session
        sess = tf.Session()
        # Init variables
        sess.run(tf.global_variables_initializer())
        # Train model
        outs_train = []
        for epoch in range(FLAGS.epochs):
            # epoch_t = time.time()

            # Construct feed dictionary
            feed_dict = construct_feed_dict(features, support, y_train1, train_mask1, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            # Training step
            outs_train = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.predict()], feed_dict=feed_dict)
            # Validation
            feed_dict_val = construct_feed_dict(features, support, y_val1, val_mask1, placeholders)
            outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
            # val_cost0 = outs_val[0]
            # val_acc0 = outs_val[1]
            # duration0 = time.time() - model_t
            # print("Validation: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(val_cost0),
            # "accuracy=", "{:.5f}".format(val_acc0), "time=", "{:.5f}".format(duration0))

        # Testing
        feed_dict_test = construct_feed_dict(features, support, y_test1, test_mask1, placeholders)
        outs_test = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_test)
        test_cost0 = outs_test[0]
        test_acc0 = outs_test[1]
        duration0 = time.time() - model_t
        # print("Testing: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(test_cost0),
        # "accuracy=", "{:.5f}".format(test_acc0), "time=", "{:.5f}".format(duration0))
        # print("Optimization Finished!")
        # print("Optimization Finished!")
        return test_cost0, test_acc0, duration0, outs_train


    #  Active Learning
    num_train_nodes = len(pool_y_index[0])
    batch = 20
    round_num = len(pool_y_index)
    num_pool_nodes = int(num_train_nodes / 2)
    # num_pool_nodes = 200
    maxIter = int(num_pool_nodes / batch)
    if maxIter > 40:
        maxIter = 40
    results = []
    model_times = []
    select_times = []
    rewards_centrality = []
    rewards_entropy = []
    rewards_density = []
    for run in range(1):
        result_temp = []
        model_time_temp = []
        select_time_temp = []

        y_all = np.zeros((all_node_num, class_num))
        y_all[all_y_index, :] = all_y_label

        val_idx = pool_y_index[run][num_pool_nodes:num_train_nodes]
        val_mask = sample_mask(val_idx, all_node_num)
        y_val = np.zeros((all_node_num, class_num))
        y_val[val_mask, :] = y_all[val_mask, :]
        pool_idx = pool_y_index[run][0:num_train_nodes]
        test_idx = test_y_index[run]
        pool_mask = sample_mask(pool_idx, all_node_num)
        test_mask = sample_mask(test_idx, all_node_num)
        y_pool = np.zeros((all_node_num, class_num))
        y_test = np.zeros((all_node_num, class_num))
        y_pool[pool_mask, :] = y_all[pool_mask, :]
        y_test[test_mask, :] = y_all[test_mask, :]
        pool_idx = pool_idx.tolist()
        random.shuffle(pool_idx)
        outs_train = []
        train_idx = []
        outs_new = []
        outs_old = []
        rewards = dict()
        reward_centrality = []
        reward_entropy = []
        reward_density = []
        rewards['centrality'] = reward_centrality
        rewards['entropy'] = reward_entropy
        rewards['density'] = reward_density
        idx_select = []
        idx_select_centrality = []
        idx_select_entropy = []
        idx_select_density = []
        dominates = dict()
        dominates['centrality'] = 0
        dominates['entropy'] = 0
        dominates['density'] = 0
        for iter_num in range(maxIter):
            select_t = time.time()
            if iter_num == 0:
                idx_select = pool_idx[0:batch]
            else:
                print("outs_train size:\t", len(outs_train[3]), len(outs_train[4]))
                idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates = \
                active_select(outs_train, old_adj, pool_idx, all_node_num, batch, importance, degree, rewards,
                              class_num, iter_num, dominates)
            select_duration = time.time() - select_t
            print("index length:\t", len(idx_select))
            print("idx_select:\t", idx_select)
            pool_idx = list(set(pool_idx) - set(idx_select))
            train_idx = train_idx + idx_select
            train_mask = sample_mask(train_idx, all_node_num)
            y_train = np.zeros((all_node_num, class_num))
            y_train[train_mask, :] = y_all[train_mask, :]
            print("train_idx_len:\t", len(train_idx))
            print("train_idx_len:\t", train_idx)
            test_cost, test_acc, model_duration, outs_train = DHNE_train(y_train, train_mask, y_val, val_mask, y_test, test_mask)
            print("dataset=" + data_str, " round=" + str(run), " iter=" + str(iter_num), "  Test set results:", "cost=",
                  "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(model_duration))

            outs_old = outs_new
            outs_new = outs_train
            if iter_num == 0:
                reward_centrality = rewards['centrality']
                reward_entropy = rewards['entropy']
                reward_density = rewards['density']
                reward_centrality.append(1)
                reward_entropy.append(1)
                reward_density.append(1)
                rewards['centrality'] = reward_centrality
                rewards['entropy'] = reward_entropy
                rewards['density'] = reward_density
            else:
                rewards = measure_rewards(outs_new, outs_old, rewards, old_adj, idx_select,
                                          idx_select_centrality, idx_select_entropy, idx_select_density)

    print("END")
