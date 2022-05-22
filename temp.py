# -*- coding: utf-8 -*-

import time

from utils2 import *
import random
from Active_select2 import *
from rewards import *

dataset = 'MovieLens'
node_objects, features, network_objects, all_y_index, all_y_label, \
pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(dataset)
importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
features = sparse_to_tuple(features)

# node_objects:每个node在dictionary里的key, 一共有28491个node, key range 0: 28490
# all_node_num = 28491
# class_num = 3, movieLens是三分类，分成三种genre

# network_objects
# all_y_index, all_y_label
# pool_y_index, test_y_index
# new_adj, old_adj
# features

# num_train_nodes = len(pool_y_index[0])
# batch = 20
# round_num = len(pool_y_index)
# num_pool_nodes = int(num_train_nodes / 2)
# # num_pool_nodes = 200
# maxIter = int(num_pool_nodes / batch)
# if maxIter > 40:
#     maxIter = 40
# results = []
# model_times = []
# select_times = []
# rewards_centrality = []
# rewards_entropy = []
# rewards_density = []
# for run in range(1):
#     result_temp = []
#     model_time_temp = []
#     select_time_temp = []

#     y_all = np.zeros((all_node_num, class_num))
#     y_all[all_y_index, :] = all_y_label

#     val_idx = pool_y_index[run][num_pool_nodes:num_train_nodes]
#     val_mask = sample_mask(val_idx, all_node_num)
#     y_val = np.zeros((all_node_num, class_num))
#     y_val[val_mask, :] = y_all[val_mask, :]
#     pool_idx = pool_y_index[run][0:num_train_nodes]
#     test_idx = test_y_index[run]
#     pool_mask = sample_mask(pool_idx, all_node_num)
#     test_mask = sample_mask(test_idx, all_node_num)
#     y_pool = np.zeros((all_node_num, class_num))
#     y_test = np.zeros((all_node_num, class_num))
#     y_pool[pool_mask, :] = y_all[pool_mask, :]
#     y_test[test_mask, :] = y_all[test_mask, :]
#     pool_idx = pool_idx.tolist()
#     random.shuffle(pool_idx)
#     outs_train = []
#     train_idx = []
#     outs_new = []
#     outs_old = []
#     rewards = dict()
#     reward_centrality = []
#     reward_entropy = []
#     reward_density = []
#     rewards['centrality'] = reward_centrality
#     rewards['entropy'] = reward_entropy
#     rewards['density'] = reward_density
#     idx_select = []
#     idx_select_centrality = []
#     idx_select_entropy = []
#     idx_select_density = []
#     dominates = dict()
#     dominates['centrality'] = 0
#     dominates['entropy'] = 0
#     dominates['density'] = 0
#     for iter_num in range(maxIter):
#         select_t = time.time()
#         if iter_num == 0:
#             idx_select = pool_idx[0:batch]
#         else:
#             idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates = \
#             active_select(outs_train, old_adj, pool_idx, all_node_num, batch, importance, degree, rewards,
#                             class_num, iter_num, dominates)
#         select_duration = time.time() - select_t

#         pool_idx = list(set(pool_idx) - set(idx_select))
#         train_idx = train_idx + idx_select
#         train_mask = sample_mask(train_idx, all_node_num)
#         y_train = np.zeros((all_node_num, class_num))
#         y_train[train_mask, :] = y_all[train_mask, :]
#         print(y_train)
# print("node objects:\n", node_objects)
# for data_index in range(len(dataset_arr)):
#     data_str = dataset_arr[data_index]
#     # print(type(data_str))
#     # print(data_str)
#     # Load data
#     node_objects, features, network_objects, all_y_index, all_y_label, \
#     pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(data_str)

#     # Some preprocessing
#     # features = preprocess_features(features)
#     features = sparse_to_tuple(features)
#     # support = [preprocess_adj(new_adj)]
#     support = []
#     for index in range(len(network_objects)):
#         adj = network_objects[index]
#         support.append(preprocess_adj(adj))
#     num_supports = len(support)
