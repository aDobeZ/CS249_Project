import numpy as np
import time
import torch as th
import torch.nn.functional as F
from model import EntityClassify
import sys
np.set_printoptions(threshold=sys.maxsize)
import copy

def RGCN_train(args, train_idx, val_idx, test_idx, labels, g, num_classes):
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

    # define parameter 
    if args.dataset == 'movielens':
        category = "movie"
        others = ['movie', 'director', 'writer', 'tag', 'user']
    elif args.dataset == 'cora':
        category = "paper"
        others = ['author', 'paper', 'term']
    elif args.dataset == 'dblp':
        category = "author"
        others = ['author', 'conf', 'paper', 'term']
    else:
        raise ValueError()
    print("others:\t", others)

    # optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
    # training loop
    print("start training...")
    dur = []
    record = []
    model.train()
    best_model = None
    best_loss = 100000000
    for epoch in range(args.n_epochs):
        optimizer.zero_grad()
        t0 = time.time()
        temp = model()
        logits = temp[category]
        # for index, cat_name in enumerate(others):
        #     if index == 0:
        #         logits = temp[cat_name]
        #     else:
        #         logits = th.cat((logits, temp[cat_name]), 0)
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
        test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
        test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
        # print("Epoch {:05d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Test Acc: {:.4f} | Test loss: {:.4f} | Time: {:.4f}".
        #       format(epoch, train_acc, loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item(), np.average(dur)))
        record.append([train_acc, loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item()])
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = copy.deepcopy(model)
    print()
    if args.model_path is not None:
        th.save(best_model.state_dict(), args.model_path)

    best_model.eval()
    result = best_model.forward()
    logits = result[category]
    # for cat_name in others: 
    #     logits = th.cat((logits, result[cat_name]), 0)
    train_acc = th.sum(logits[train_idx].argmax(dim=1) == labels[train_idx]).item() / len(train_idx)
    val_loss = F.cross_entropy(logits[val_idx], labels[val_idx])
    val_acc = th.sum(logits[val_idx].argmax(dim=1) == labels[val_idx]).item() / len(val_idx)
    test_acc = th.sum(logits[test_idx].argmax(dim=1) == labels[test_idx]).item() / len(test_idx)
    test_loss = F.cross_entropy(logits[test_idx], labels[test_idx])
    best_result = [train_acc, loss.item(), val_acc, val_loss.item(), test_acc, test_loss.item()]
    print("Best Model -- Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    new_logits = result[others[0]]
    # 对于cora缺少edge的特殊处理
    if args.dataset == 'cora':
        new_logits = th.cat((new_logits, th.zeros(7550, num_classes)), 0)
    print("cat_name:\t", others[0], "\t shape:\t", new_logits.shape)
    for index in range(1, len(others)):
        new_logits = th.cat((new_logits, result[others[index]]), 0)
        print("cat_name:\t", others[index], "\t shape:\t", new_logits.shape)

    print("new_logits_shape:", new_logits.shape)
    print()
    return new_logits, record, best_result
