import argparse
from utils.map_calculate import *
from model.model import TestModel
from utils.load_data import eval_load

parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--dataset', default='flickr',
                    type=str, help=' ')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for training')
parser.add_argument('--hash_code_length', default=32,
                    type=int, help='hash-code-length')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--num_classes', default=38, type=int,
                    help='Number of class')
parser.add_argument('--weight_pth', default='', type=str,
                    help='model weight path')

args = parser.parse_args()

net = TestModel(num_classes=args.num_classes, hash_code_length=args.hash_code_length)

data_tmp = torch.load(args.weight_pth, map_location="cuda:0")
data_tmp = {k.lstrip("module."): v for k, v in data_tmp.items()}
net.load_state_dict(data_tmp)
net = net.cuda()
net.eval()

eval_loader, database_loader = eval_load(args)

test_label_matrix = np.empty(shape=(0, 22))
test_hash_matrix = np.empty(shape=(0, 64))
for ii, batch_iterator in enumerate(tqdm(eval_loader)):
    with torch.no_grad():
        images, labels, boxes = batch_iterator
        test_label_matrix = np.concatenate((test_label_matrix, labels), axis=0)

        images = images.cuda()
        labels = labels.cuda()
        boxes = boxes.cuda()
        out2, out_class = net(images, boxes)

        hash_code = torch.sign(out2)
        hash_code = hash_code.cpu().numpy()
        test_hash_matrix = np.concatenate((test_hash_matrix, hash_code), axis=0)

database_label_matrix = np.empty(shape=(0, 22))
database_hash_matrix = np.empty(shape=(0, 64))
for ii, batch_iterator in enumerate(tqdm(database_loader)):
    with torch.no_grad():
        images, labels, boxes = batch_iterator
        database_label_matrix = np.concatenate((database_label_matrix, labels), axis=0)

        images = images.cuda()
        labels = labels.cuda()
        boxes = boxes.cuda()
        out2, out_class = net(images, boxes)

        hash_code = torch.sign(out2)
        hash_code = hash_code.cpu().numpy()
        database_hash_matrix = np.concatenate((database_hash_matrix, hash_code), axis=0)

acg100, ndcg100 = acg_test(database_hash_matrix, test_hash_matrix, database_label_matrix,
                                           test_label_matrix, 100)
print(acg100, ndcg100)


