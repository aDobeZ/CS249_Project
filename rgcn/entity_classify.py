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
np.set_printoptions(threshold=sys.maxsize)

from os.path import dirname, abspath, join
import sys
ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = join(ROOT_PATH, "data")
sys.path.insert(0, ROOT_PATH)
from data import movielens_loader, cora_loader

def main(args):
    # load graph data
    if args.dataset == 'movielens':
        dataloader = movielens_loader
        category = "movie"
    elif args.dataset == 'cora':
        dataloader = cora_loader
        category = "paper"
    else:
        raise ValueError()

    # To support movielens dataset
    g, all_y_index, all_y_label, train_y_index, test_y_index, num_classes = dataloader(DATA_PATH)
    pool_index = np.array(train_y_index[0])
    num_train_nodes = len(pool_index)    
    train_idx = th.from_numpy(pool_index[0:int(num_train_nodes/2)])
    val_idx = th.from_numpy(pool_index[int(num_train_nodes/2):num_train_nodes])
    test_idx = th.from_numpy(np.array(test_y_index[0]))

    labels = th.from_numpy(all_y_label)

    # define train index 

    # category_id = len(g.ntypes)
    # for i, ntype in enumerate(g.ntypes):
    #     if ntype == category:
    #         category_id = i

    # # split dataset into train, validate, test
    # if args.validation:
    #     val_idx = train_idx[:len(train_idx) // 5]
    #     train_idx = train_idx[len(train_idx) // 5:]
    # else:
    #     val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)
        g = g.to('cuda:%d' % args.gpu)
        labels = labels.cuda()
        train_idx = train_idx.cuda()
        val_idx = val_idx.cuda()
        test_idx = test_idx.cuda()

    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        if epoch > 5:
            t0 = time.time()
        temp = model()
        logits = temp[category]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        if epoch > 5:
            dur.append(t1 - t0)
        # print("train_index:\t", train_idx)
        # print("val_idx:\t", val_idx)
        print(len(train_idx), len(val_idx))
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    model.eval()
    logits = model.forward()[category]
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
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
