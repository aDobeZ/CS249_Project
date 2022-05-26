"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import EntityClassify
import sys
from RGCN_train import RGCN_train
np.set_printoptions(threshold=sys.maxsize)
from os.path import dirname, abspath, join

DATA_PATH = join(dirname(dirname(abspath(__file__))), "data")
sys.path.insert(0, join(DATA_PATH, 'MovieLens'))
from data_loader import process_movielens as movielens_loader

def main(args):
    # load graph data
    if args.dataset == 'movielens':
        dataloader = movielens_loader
    else:
        raise ValueError()

    # To support movielens dataset
    g, all_y_index, all_y_label, train_y_index, test_y_index = dataloader(DATA_PATH)
    category = "movie"
    num_classes = 3
    pool_index = np.array(train_y_index[0])
    num_train_nodes = len(pool_index)    
    train_idx = th.from_numpy(pool_index[0:int(num_train_nodes/2)])
    val_idx = th.from_numpy(pool_index[int(num_train_nodes/2):num_train_nodes])
    test_idx = th.from_numpy(np.array(test_y_index[0]))

    labels = th.from_numpy(all_y_label)

    # preprocess train index 

    category_id = len(g.ntypes)
    for i, ntype in enumerate(g.ntypes):
        if ntype == category:
            category_id = i
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
            logits = RGCN_train(args, train_idx, val_idx, test_idx, labels, g)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=3,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
