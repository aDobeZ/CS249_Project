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
import copy
np.set_printoptions(threshold=sys.maxsize)

from os.path import dirname, abspath, join
import sys
ROOT_PATH = dirname(dirname(abspath(__file__)))
DATA_PATH = join(ROOT_PATH, "data")
sys.path.insert(0, ROOT_PATH)
from data import movielens_loader, cora_loader

def RGCN_baseline(args, train_idx, val_idx, test_idx, labels, g, num_classes):
    # load graph data
    if args.dataset == 'movielens':
        dataloader = movielens_loader
        category = "movie"
    elif args.dataset == 'cora':
        dataloader = cora_loader
        category = "paper"
    else:
        raise ValueError()

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
                           dropout=0,
                           use_self_loop=args.use_self_loop)
    
    best_loss = 10000000
    best_model = None

    if use_cuda:
        model.cuda()

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    model.train()
    record = []
    for epoch in range(50):
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
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        # print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
        #       format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
        record.append([train_acc, loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item()])
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
    print()
    if args.model_path is not None:
        th.save(best_model.state_dict(), args.model_path)

    best_model.eval()
    logits = best_model.forward()[category]
    train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    best_result = [train_acc, loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item()]
    print("Best Model -- Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()
    return record, best_result