import os
import torch
import torch.optim as optim
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from model.model import TestModel
from torchsummary import summary
from utils.load_data import data_load, eval_load
from loss.loss import HashLoss
from utils.map_calculate import acg_test, model_eval


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--dataset', default='flickr',
                    type=str, help=' ')
parser.add_argument('--batch_size', default=256, type=int,
                    help='Batch size for training')
parser.add_argument('--max_epoch', default=1500, type=int,
                    help='Max Epoch for training')
parser.add_argument('--hash_code_length', default=32,
                    type=int, help='hash-code-length')
parser.add_argument('--num_workers', default=10, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--num_classes', default=38, type=int,
                    help='Number of class')
parser.add_argument('--eval_interval', default=15, type=int,
                    help='Number of eval')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--eps', default=1e-8, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--alpha', default=1., type=float,
                    help='classfication loss hyperparameter')
parser.add_argument('--beta', default=1e-4, type=float,
                    help='quanti loss hyperparameter')
parser.add_argument('--beta1', default=1e-2, type=float,
                    help='hash loss hyperparameter')

parser.add_argument('--save_path', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--num_gpus', default=1, type=int)

args = parser.parse_args()

def train():

    if torch.cuda.is_available():
        torch.cuda.manual_seed(18)
    else:
        torch.manual_seed(18)

    pth_path = os.path.join(args.save_path, args.dataset)
    runing_path = pth_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    weight_path = runing_path + f'/checkpoint/'
    os.makedirs(weight_path, exist_ok=True)


    train_loader, test_loader = data_load(args)
    eval_loader, database_loader = eval_load(args)

    net = TestModel(num_classes=args.num_classes, hash_code_length=args.hash_code_length)

    if args.cuda:
        if args.num_gpus > 1:
            net = torch.nn.DataParallel(net, device_ids=[0, 1])
        net = net.cuda()
        torch.backends.cudnn.benchmark = True

    summary(net, input_size=[(27, 2048), (27, 4)])

    net.train()

    optimizer = optim.Adam(net.parameters(), lr=args.lr, eps=args.eps)
    Criterion = HashLoss(args.num_classes, args.hash_code_length)


    step = 0
    best_acg = 0.0
    try:
        for epoch in range(args.max_epoch):
            progress_bar = tqdm(train_loader)
            for ii, batch_iterator in enumerate(progress_bar):
                images, labels, boxes = batch_iterator
                if args.cuda:
                    images = images.cuda()
                    labels = labels.cuda()
                    boxes = boxes.cuda()

                out2, out_class = net(images, boxes.float())
                optimizer.zero_grad()
                classify_loss, hash_loss, quanti_loss = Criterion(out2, out_class, labels)
                loss = args.alpha * classify_loss + args.beta1 * hash_loss + args.beta * quanti_loss

                loss.backward()
                optimizer.step()

                progress_bar.set_description(
                    'Step:{}.Epoch:{}/{}. Hash loss:{:.5f}. Quantization loss:{:.5f}. '
                    'Total loss:{:.5f}.'.format(
                        step, epoch, args.max_epoch,
                        hash_loss.item(),
                        quanti_loss.item(), loss))

                step = step + 1

            net.eval()
            eval_loss = []
            eval_cls_loss = []
            eval_hash_loss = []
            eval_quanti_loss = []
            for ii, batch_iterator in enumerate(test_loader):
                with torch.no_grad():
                    images, labels, boxes = batch_iterator
                    if args.cuda:
                        images = images.cuda()
                        labels = labels.cuda()
                        boxes = boxes.cuda()
                    out2, out_class = net(images, boxes.float())

                    classify_loss, hash_loss, quanti_loss = Criterion(out2, out_class, labels)
                    loss = args.alpha * classify_loss + args.beta1 * hash_loss + args.beta * quanti_loss

                    if images.shape[0] == args.batch_size:
                        eval_loss.append(loss.item())
                        eval_cls_loss.append(classify_loss.item())
                        eval_hash_loss.append(hash_loss.item())
                        eval_quanti_loss.append(quanti_loss.item())


            print(
                'Val. Epoch: {}/{}. Hash loss: {:.5f}. Quanti loss: {:.5f}. '
                'Total loss: {:.5f}.'.format(
                    epoch, args.max_epoch, np.mean(eval_hash_loss).item(),
                    np.mean(eval_quanti_loss).item(), np.mean(eval_loss).item()))

            if epoch % args.eval_interval == 0:
                os.makedirs(weight_path + '/hash_code/', exist_ok=True)
                test_hash_matrix, test_label_matrix, database_hash_matrix, database_label_matrix = model_eval(net,
                                                                                                              eval_loader,
                                                                                                              database_loader,
                                                                                                              args)
                acg100, ndcg100 = acg_test(database_hash_matrix, test_hash_matrix, database_label_matrix,
                                           test_label_matrix, 100)
                if acg100 > best_acg:
                    best_acg = acg100
                    best_ndcg = ndcg100
                    np.savez(
                        weight_path + '/hash_code/acg_' + str(best_acg) + '_test_hash_label',
                        label=test_label_matrix, hash_code=test_hash_matrix)
                    np.savez(
                        weight_path + '/hash_code/acg_' + str(best_acg) + '_database_hash_label',
                        label=database_label_matrix, hash_code=database_hash_matrix)
                    save_weights = os.path.join(weight_path, 'model_best.pth')
                    torch.save(net.state_dict(), save_weights)

                print("acg:%.3f, Best acg: %.3f, ndcg:%.3f" % (
                    acg100, best_acg, ndcg100
                ))

            net.train()

            if epoch % args.eval_interval == 0 or epoch == args.max_epoch - 1:
                torch.save(net.state_dict(), os.path.join(weight_path, f'{args.dataset}_{epoch}.pth'))

    finally:
        torch.save(net.state_dict(), os.path.join(weight_path, f'{args.dataset}_last.pth'))


if __name__ == '__main__':
    print(args)
    train()
