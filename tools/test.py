from yaml import loader
import _init_paths
import yaml
import argparse
import numpy as np
import os.path as osp
from attrdict import AttrDict

import torch
from torch.utils.data import DataLoader

from datasets.mini_imageNet import MiniImageNet
from utils.samplers import CategoriesSampler
from model.convnet import Convnet
from utils.utils import pprint, set_gpu, count_acc, Averager, euclidean_metric, cosine, copyModel, CI, ensure_path, setup_seed



def initConfig():
    with open(args.config, 'r') as f:
        cfg = AttrDict(yaml.load(f))

    ensure_path(osp.join(cfg.save_path, "result"))
    if args.seed != None:
        setup_seed(args.seed)
    else:
        setup_seed(cfg.seed)

    set_gpu(cfg.test.gpu)

    return cfg

def getDataloader():

    dataset = MiniImageNet(cfg.datapath, 'test')
    sampler = CategoriesSampler(dataset.label,
                                cfg.test.batch, cfg.test.way, cfg.test.shot + cfg.test.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    return loader

def loadmodel():
    checkpoint = torch.load(osp.join(cfg.save_path, cfg.test.load))
    model = copyModel(checkpoint["model"], Convnet()).cuda()

    return model

def test(model, loader):
    model.eval()

    allacc = []
    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        p = cfg.test.shot * cfg.test.way
        data_shot, data_query = data[:p], data[p:]

        proto = model(data_shot)
        proto = proto.reshape(cfg.test.shot, cfg.test.way, -1).mean(dim=0)

        logits = euclidean_metric(model(data_query), proto)

        label = torch.arange(cfg.test.way).repeat(cfg.test.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

        allacc.append(acc)
        
        p = None; proto = None; logits = None
        
    allacc = np.array(allacc)
    torch.save(allacc, osp.join(cfg.save_path, "result/allacc"))

    return allacc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default="exps/exp-v1/config.yaml")
    parser.add_argument('--seed', default=None)

    args = parser.parse_args()
    pprint(vars(args))

    cfg = initConfig()

    loader = getDataloader()

    model = loadmodel()

    allacc = test(model, loader)

    mean, std, conf_intveral = CI(allacc)

    result = "mean: " + str(mean) + "\nstd: " + str(std) + "\nconfidence intveral: [" + str(conf_intveral[0]) + " : " + str(conf_intveral[1]) + "]"

    with open( osp.join(cfg.save_path, "result/acc.txt"), 'w') as f:
        f.write(result)