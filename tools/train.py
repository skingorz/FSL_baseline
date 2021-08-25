import _init_paths
import argparse
import yaml
import os.path as osp
from attrdict import AttrDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, copyModel, setup_seed


def getDataloader():

    trainset = MiniImageNet(cfg.datapath, 'train')
    train_sampler = CategoriesSampler(trainset.label, 100,
                                      cfg.train.train_way, cfg.train.shot + cfg.train.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=16, pin_memory=True)

    valset = MiniImageNet(cfg.datapath, 'val')
    val_sampler = CategoriesSampler(valset.label, 400,
                                    cfg.train.test_way, cfg.train.shot + cfg.train.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=16, pin_memory=True)

    return train_loader, val_loader


def initConfig():
    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    if not args.resume:
        ensure_path(cfg.save_path)
        if args.seed != None:
            setup_seed(args.seed)
        else:
            setup_seed(cfg.seed)

    set_gpu(cfg.train.gpu)

    trlog = {}
    trlog['args'] = vars(args)
    trlog['cfg'] = vars(cfg)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    return cfg, trlog

    

def initTrain(isResume=False):

    model = nn.DataParallel(Convnet()).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    start_epoch = 1

    if isResume:
        checkpoint = torch.load(osp.join(cfg.save_path, "epoch-last.pth"))
        model = copyModel(checkpoint["model"], nn.DataParallel(Convnet())).cuda()
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_schedule"])
        start_epoch = checkpoint["epoch"] + 1
    
    return model, optimizer, lr_scheduler, start_epoch

def train_epoch(cfg, model, optimizer, epoch, train_loader):
    model.train()

    tl = Averager()
    ta = Averager()

    for i, batch in enumerate(train_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = cfg.train.shot * cfg.train.train_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(cfg.train.shot, cfg.train.train_way, -1).mean(dim=0)

        label = torch.arange(cfg.train.train_way).repeat(cfg.train.query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)
        print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                .format(epoch, i, len(train_loader), loss.item(), acc))

        tl.add(loss.item())
        ta.add(acc)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        p = None; proto = None; logits = None; loss = None

    tl = tl.item()
    ta = ta.item()

    return tl, ta

def eval(cfg, model, val_loader):

    model.eval()

    vl = Averager()
    va = Averager()


    for i, batch in enumerate(val_loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = cfg.train.shot * cfg.train.test_way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(cfg.train.shot, cfg.train.test_way, -1).mean(dim=0)

        label = torch.arange(cfg.train.test_way).repeat(cfg.train.query)
        label = label.type(torch.cuda.LongTensor)

        logits = euclidean_metric(model(data_query), proto)
        loss = F.cross_entropy(logits, label)
        acc = count_acc(logits, label)

        vl.add(loss.item())
        va.add(acc)
        
        p = None; proto = None; logits = None; loss = None


    vl = vl.item()
    va = va.item()

    return vl, va

def train(cfg, model, optimizer, lr_scheduler, train_loader, val_loader, start_epoch=1):

    def save_model(name):
        checkpoint = {
            "model": model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            'lr_schedule': lr_scheduler.state_dict()
        }
        torch.save(checkpoint, osp.join(cfg.save_path, name + '.pth'))
        # torch.save(model.module.state_dict(), osp.join(cfg.save_path, name + '.pth'))

    timer = Timer()

    for epoch in range(start_epoch, cfg.train.max_epoch + 1):
        # torch.cuda.empty_cache()
        lr_scheduler.step()

        tl, ta = train_epoch(cfg, model, optimizer, epoch, train_loader)
        vl, va = eval(cfg, model, val_loader)

        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(cfg.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % cfg.train.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure((epoch-start_epoch + 1) / (cfg.train.max_epoch-start_epoch))))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v1/config.yaml")
    parser.add_argument('--seed', default=None)
    parser.add_argument('--resume', action="store_true", default=False)

    args = parser.parse_args()
    pprint(vars(args))

    cfg, trlog = initConfig()
    model, optimizer, lr_scheduler, start_epoch = initTrain(args.resume)
    train_loader, val_loader = getDataloader()

    train(cfg, model, optimizer, lr_scheduler, train_loader, val_loader, start_epoch)
