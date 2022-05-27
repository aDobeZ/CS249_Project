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
from active_select_RGCN import *
from rewards_RGCN import *
from utils2 import *
from RGCN_baseline import *
np.set_printoptions(threshold=sys.maxsize)
from os.path import dirname, abspath, join

from os.path import dirname, abspath, join
import sys
ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = join(ROOT_PATH, "data")
sys.path.insert(0, ROOT_PATH)
from data import movielens_loader, cora_loader

def main(args):

    # load graph data
	if args.dataset == "movielens":
		dataloader = movielens_loader
		category = "movie"
		dataset = 'MovieLens'
		min_index = 0
		batch = 20
	elif args.dataset == 'cora':
		dataloader = cora_loader
		category = "paper"
		dataset = 'Cora'
		min_index = 24961
		batch = 80
	else:
		raise ValueError()

	# To support movielens dataset
	g, all_y_index, all_y_label, train_y_index, test_y_index, num_classes = dataloader(DATA_PATH)
	pool_index = np.array(train_y_index[0])
	num_train_nodes = len(pool_index)
	num_pool_nodes = int(num_train_nodes / 2)
	# train_idx = th.from_numpy(pool_index[0:num_pool_nodes])
	val_idx = th.from_numpy(pool_index[num_pool_nodes:num_train_nodes])
	test_idx = th.from_numpy(np.array(test_y_index[0]))
	pool_index = pool_index[0:num_pool_nodes].tolist()
	for index, value in enumerate(pool_index):
		pool_index[index] += min_index

	labels = th.from_numpy(all_y_label)
	soft_function = nn.Softmax(dim=1)
	
	# generate needed parameters
	node_objects, features, network_objects, all_y_index, all_y_label, \
	pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(dataset)
	importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
	print("adj_shape:\t", old_adj.shape)

	# preprocess train index
	maxIter = int(num_pool_nodes / batch)
	if maxIter > 40: 
		maxIter = 40
		maxIter = 5

	# define parameters
	outs_train = []
	train_idx = []
	rgcn_idx = []
	outs_new = []
	outs_old = []
	rewards = {'centrality': [1], 'entropy': [1], 'density': [1]}
	dominates = {'centrality': 0, 'entropy': 0, 'density': 0}
	idx_select = []
	idx_select_centrality = []
	idx_select_entropy = []
	idx_select_density = []
	record = np.zeros((1, 6))
	best_active = []
	baseline_record = np.zeros((1, 6))
	best_baseline = []
	for iter_num in range(maxIter):
		print("current iteration: \t", iter_num + 1)
		if iter_num == 0:
			idx_select = pool_index[0:batch]
		else:
			idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates = \
			active_select(outs_train, outs_new, old_adj, pool_index, all_node_num, batch, importance, degree, rewards,
							class_num, iter_num, dominates)
		# idx_select = idx_select.tolist()
		print("select index length:\t", len(idx_select))
		print("idx_select:\t", idx_select)
		pool_index = list(set(pool_index) - set(idx_select))
		train_idx += idx_select
		for index in idx_select:
			rgcn_idx.append(index - min_index)
		print("train index length:\t", len(train_idx))
		print("train index:\t", train_idx)
		print("rgcn index: \t", rgcn_idx)
		logits, new_record, best_result = RGCN_train(args, th.from_numpy(np.asarray(rgcn_idx)), val_idx, test_idx, labels, g, class_num)
		best_active.append(best_result)
		record = np.concatenate((record, np.array(new_record)), axis=0)
		outs_train = logits.detach().numpy()
		outs_old = outs_new
		outs_new = soft_function(logits)
		outs_new = outs_new.detach().numpy()
		# compute rewards after the 1st iteration
		if iter_num > 0:
			rewards = measure_rewards(outs_new, outs_old, rewards, old_adj, idx_select, idx_select_centrality, idx_select_entropy, idx_select_density)
	
	print("iteration end")
	record = record[1:]
	for iter_num in range(1, 11):
		base_record, base_best = RGCN_baseline(args, th.from_numpy(np.asarray(rgcn_idx)), val_idx, test_idx, labels, g, class_num)
		baseline_record = np.concatenate((baseline_record, np.array(base_record)), axis=0)
		best_baseline.append(base_best)
	baseline_record = baseline_record[1:]
	print("ActiveRGCN record shape:\t\t", record.shape)
	print("RGCNbaseline record shape:\t", np.array(baseline_record).shape)
	print("best active:\n", np.array(best_active))
	print("best baseline:\n", np.array(best_baseline))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=200,
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
