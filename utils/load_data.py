import numpy
import torch
import os
import numpy as np
import argparse
import h5py
import pickle
from PIL import Image
import torchvision.transforms as transforms


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, root,
                 transform=None, target_transform=None, train=True, database_bool=False, num_classes=38):
        self.root = root
        self.transform = transform
        self.num_classes = num_classes
        self.target_transform = target_transform
        if train:
            self.base_folder = 'all_train.txt'
        elif database_bool:
            self.base_folder = 'all_database.txt'
        else:
            self.base_folder = 'all_test.txt'

        self.feature_data = h5py.File(os.path.join(self.root, 'all_att.h5'), 'r')
        self.box_data = h5py.File(os.path.join(self.root, 'all_box.h5'), 'r')
        f = open(os.path.join(self.root, 'all_info.pkl'), 'rb')
        self.img_wh = pickle.load(f)

        self.train_data = []
        self.train_labels = []

        filename = os.path.join(self.root, self.base_folder)

        with open(filename, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()

                if not lines:
                    break
                pos_tmp = lines.split()[0]
                label_tmp = lines.split()[1:]
                self.train_data.append(pos_tmp)
                self.train_labels.append(label_tmp)
        self.train_data = np.array(self.train_data)
        self.train_labels = np.array(self.train_labels, dtype=np.float)
        self.train_labels.reshape((-1, self.num_classes))


    def __getitem__(self, index):

        imgname, target = self.train_data[index], self.train_labels[index]
        target = target.astype(np.float32)
        img = self.feature_data[imgname][:]
        box = self.box_data[imgname][:]
        h = self.img_wh[imgname]['image_h']
        w = self.img_wh[imgname]['image_w']

        x_min = int(min(box[:, 0]))
        y_min = int(min(box[:, 1]))
        box[:, 0] = box[:, 0] - x_min
        box[:, 2] = box[:, 2] - x_min
        box[:, 1] = box[:, 1] - y_min
        box[:, 3] = box[:, 3] - y_min
        w = w - x_min
        h = h - y_min

        box[:, 0] = box[:, 0] / w
        box[:, 2] = box[:, 2] / w
        box[:, 1] = box[:, 1] / h
        box[:, 3] = box[:, 3] / h

        return img, target, box

    def __len__(self):
        return len(self.train_data)


def data_load(args):
    # Dataset
    train_dataset = MyDataset(root=os.path.join('./data', args.dataset),
                            train=True,
                            transform=None,
                            num_classes=args.num_classes)

    test_dataset = MyDataset(root=os.path.join('./data', args.dataset),
                           train=False,
                           transform=None,
                           num_classes=args.num_classes)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)
    return train_loader, test_loader


def eval_load(args):
    # Dataset

    test_dataset = MyDataset(root=os.path.join('./data', args.dataset),
                           train=False,
                           transform=None,
                           num_classes=args.num_classes)

    database_dataset = MyDataset(root=os.path.join('./data', args.dataset),
                           train=False,
                           database_bool=True,
                           transform=None,
                           num_classes=args.num_classes)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)
    return test_loader, database_loader

