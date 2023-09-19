import argparse
import pickle
import time
from utils import *
from model import *
from madgrad import MADGRAD
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn import Module, Parameter
import torch.nn.functional as F
import datetime
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sample', help='dataset name: diginetica/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=2048, help='input batch size')
parser.add_argument('--heads', type=int, default=8, help='heads')
parser.add_argument('--dim_feedforward', type=int, default=512, help='heads')
parser.add_argument('--epochs', type=int, default=70, help='the number of epochs to train for')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout in encoder block')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.9, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=2, help='the number of steps after which the learning rate decay')
parser.add_argument('--num_layers', type=int, default=1, help='encoder layers')
parser.add_argument('--pos_encoding', type=int, default=0, help='0 for fixed encoding, 1 for trainable encoding')
parser.add_argument('--patience', type=int, default=20, help='the number of epoch to wait before early stop ')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
opt = parser.parse_args()
print(opt)


def main():
    all_train = pickle.load(open('datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))
    transe_emb = pickle.load(open('datasets/' + opt.dataset + '/transe_emb', 'rb'))
    
    train_data, test_data, embeddings, vocab = Prepare_Data(all_train,train_data, test_data,transe_emb)
    train_data = RecSysDataset(train_data)
    test_data = RecSysDataset(test_data)

    train_loader = DataLoader(train_data, batch_size = opt.batchSize, shuffle = True)
    test_loader = DataLoader(test_data, batch_size = opt.batchSize, shuffle = False)

    model = Transformer(
    embeddings,
    nhead=opt.heads,
    dim_feedforward=opt.dim_feedforward, 
    num_layers=opt.num_layers,
    dropout=opt.dropout,
    pos_encoding = opt.pos_encoding,
).to(device)

    criterion = nn.CrossEntropyLoss()
    
    # optimizer = torch.optim.Adam(
    # (p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=1e-9)

    optimizer = MADGRAD(model.parameters(), lr=opt.lr, weight_decay=1e-9)
    scheduler = StepLR(optimizer, step_size = opt.lr_dc_step, gamma = opt.lr_dc, verbose=True)
    torch.manual_seed(0)


    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epochs):
        # train for one epoch
        scheduler.step(epoch = epoch)
        trainForEpoch(train_loader, model, optimizer, epoch, opt.epochs , criterion, log_aggr = 100)

        hit, mrr = validate(test_loader, model)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))
    
if __name__ == '__main__':
    main()
