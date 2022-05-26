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

def RGCN_train(args, train_idx, val_idx, test_idx, labels, g):
    category = "movie"
    num_classes = 3
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
    others = ['director', 'writer', 'tag', 'user']
    # training loop
    print("start training...")
    dur = []
    model.train()
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        temp = model()
        logits = temp[category]
        for cat_name in others: 
            logits = th.cat((logits, temp[cat_name]), 0)
        # logits = logits[0: 28491]
        # print(logits.shape)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()
        t1 = time.time()

        dur.append(t1 - t0)
        train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
        val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
        val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
        print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), np.average(dur)))
    print()
    # if args.model_path is not None:
    #     th.save(model.state_dict(), args.model_path)

    model.eval()
    result = model.forward()
    logits = result[category]
    for cat_name in others: 
        logits = th.cat((logits, result[cat_name]), 0)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print("logits_shape:", logits.shape)
    print()
    return logits
