# -*- coding: utf-8 -*-

import time

from utils2 import *
import random
from Active_select2 import *
from rewards import *

import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

dataset = 'MovieLens'
node_objects, features, network_objects, all_y_index, all_y_label, \
pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(dataset)
importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
features = sparse_to_tuple(features)
support = []
for index in range(len(network_objects)):
    adj = network_objects[index]
    support.append(preprocess_adj(adj))
num_supports = len(support)

# node_objects:每个node在dictionary里的key, 一共有28491个node, key range 0: 28490
# all_node_num = 28491
# class_num = 3, movieLens是三分类，分成三种genre
# network_objects: list of adjacency matrix, 此处为movie_director, movie_tag, movie_writer, user_movie_rating
# all_y_index: list of node index, length is 3672
# all_y_label: ndarray, shape num_class * num_node, which is 3 * 3672
# pool_y_index, test_y_index: pool_y_index is train_y_index(node index), shape 10 * 1836, similar for test_y_index, same size
# new_adj, old_adj: 四个adjacency matrix合在一起, old_adj都是01, new_adj有float是rating
# features: tuple, first value is 28491 * 2 ndarray, repeat each node twice, second is np.ones(node_num)
# importance: ndarray, 28491 * 1, represents importance of each node, importance具体怎么算看Active_select 2
# degree: ndarray, 28491 * 1, represents degree of each node

#  Active Learning
num_train_nodes = len(pool_y_index[0]) # 1836
batch = 20
round_num = len(pool_y_index) # 10 rounds
num_pool_nodes = int(num_train_nodes / 2) # 918
# num_pool_nodes = 200
maxIter = int(num_pool_nodes / batch) # 45.9
if maxIter > 40:
    maxIter = 40
    maxIter = 1 #这行回头记得comment掉
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
    y_all[all_y_index, :] = all_y_label # 28491 * 3, 28491里面只label了3672个

    val_idx = pool_y_index[run][num_pool_nodes:num_train_nodes] # pool_y_index[0][918:1836]后半部分
    val_mask = sample_mask(val_idx, all_node_num) # validationn node index = 1, 其余为0, node_num * 1 ndarray
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
    print(y_test.shape, test_mask.shape)
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
        else: # 这行会报错，因为把下面training过程删了
            idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates = \
            active_select(outs_train, old_adj, pool_idx, all_node_num, batch, importance, degree, rewards,
                            class_num, iter_num, dominates) 
        select_duration = time.time() - select_t

        pool_idx = list(set(pool_idx) - set(idx_select))
        train_idx = train_idx + idx_select
        train_mask = sample_mask(train_idx, all_node_num)
        y_train = np.zeros((all_node_num, class_num))
        y_train[train_mask, :] = y_all[train_mask, :]
# y_train
# train_mask
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

