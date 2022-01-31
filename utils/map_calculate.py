import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def mean_average_precision_R(database_hash, test_hash, database_labels, test_labels, R):

    one_hot_database = database_labels
    one_hot_test = test_labels

    if R == -1:
        R = database_hash.shape[0]
    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    APx = []
    Recall = []

    for i in tqdm(range(query_num)):  # for i=0
        label = one_hot_test[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch_acg = np.sum(one_hot_database[idx[0:R], :] == label, axis=1)
        imatch = imatch_acg > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)   # 累加
        Px = Lx.astype(float) / np.arange(1, R + 1, 1)

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(one_hot_database == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall))


def model_eval(net, eval_loader, database_loader, args):
    test_label_matrix = np.empty(shape=(0, args.num_classes))
    test_hash_matrix = np.empty(shape=(0, args.hash_code_length))
    for ii, batch_iterator in enumerate(tqdm(eval_loader)):
        with torch.no_grad():
            images, labels, boxes = batch_iterator
            test_label_matrix = np.concatenate((test_label_matrix, labels), axis=0)
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                boxes = boxes.cuda()
            out2, out_class = net(images, boxes.float())

            hash_code = torch.sign(out2)
            hash_code = hash_code.cpu().numpy()
            test_hash_matrix = np.concatenate((test_hash_matrix, hash_code), axis=0)

    database_label_matrix = np.empty(shape=(0, args.num_classes))
    database_hash_matrix = np.empty(shape=(0, args.hash_code_length))
    for ii, batch_iterator in enumerate(tqdm(database_loader)):
        with torch.no_grad():
            images, labels, boxes = batch_iterator
            database_label_matrix = np.concatenate((database_label_matrix, labels), axis=0)
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                boxes = boxes.cuda()
            out2, out_class = net(images, boxes.float())

            hash_code = torch.sign(out2)
            hash_code = hash_code.cpu().numpy()
            database_hash_matrix = np.concatenate((database_hash_matrix, hash_code), axis=0)

    return test_hash_matrix, test_label_matrix, database_hash_matrix, database_label_matrix


def acg_test(database_hash, test_hash, database_labels, test_labels, n):
    one_hot_database = database_labels
    one_hot_test = test_labels

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)
    ids = np.argsort(-sim, axis=0)

    ACG_all = []
    NDCG_all = []

    for i in range(query_num):  # for i=0
        # print(i)
        label = one_hot_test[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(one_hot_database[idx[0:n], :] == label, axis=1)
        imatch_all = np.sum(one_hot_database == label, axis=1)
        ACG_all.append(imatch.mean())
        DCG = (2 ** imatch - 1) / np.log2(np.arange(2, n + 2, 1))
        DCG = np.sum(DCG)

        imatch_sort = np.array(sorted(imatch_all, reverse=True))
        # imatch_sort = np.array(sorted(imatch_sort[:n]))
        DCG_max = np.sum((2 ** imatch_sort[:n] - 1) / np.log2(np.arange(2, n + 2, 1)))
        if DCG_max == 0:
            print(i)
            print(DCG)
            print(DCG_max)
            NDCG_all.append(0)
        else:
            NDCG_all.append(DCG / DCG_max)

    return np.mean(np.array(ACG_all)), np.mean(np.array(NDCG_all))
