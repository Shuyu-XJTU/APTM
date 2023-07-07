import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

from dataset.re_dataset import re_train_dataset, re_test_dataset, re_test_dataset_icfg, re_train_dataset_attr, \
    re_test_dataset_attr
from dataset.randaugment import RandomAugment
from dataset.random_erasing import RandomErasing


def create_dataset(dataset, config, evaluate=False):
    # gene
    gene_norm = transforms.Normalize((0.4416847, 0.41812873, 0.4237452), (0.3088255, 0.29743394, 0.301009))
    # cuhk
    cuhk_norm = transforms.Normalize((0.38901278, 0.3651612, 0.34836376), (0.24344306, 0.23738699, 0.23368555))
    # icfg
    icfg_norm = transforms.Normalize((0.30941582, 0.28956893, 0.30347288), (0.25849792, 0.24547698, 0.2366199))
    # rstp
    rstp_norm = transforms.Normalize((0.27722597, 0.26065794, 0.3036557), (0.2609547, 0.2508087, 0.25293276))
    # pa100k
    pa100k_norm = transforms.Normalize((0.46485138, 0.45038012, 0.4632019), (0.25088054, 0.24609283, 0.24240193))

    if dataset == 're_cuhk':
        train_norm = cuhk_norm
        test_norm = cuhk_norm
    elif dataset == 're_icfg':
        train_norm = icfg_norm
        test_norm = icfg_norm
    elif dataset == 're_rstp':
        train_norm = rstp_norm
        test_norm = rstp_norm
    elif dataset == 're_gene':
        train_norm = gene_norm
        test_norm = cuhk_norm
    elif dataset == 're_pa100k':
        train_norm = pa100k_norm
        test_norm = pa100k_norm

    train_transform = transforms.Compose([
        # transforms.RandomResizedCrop((config['h'], config['h']),
        #                              scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize',
                                              'Brightness', 'Sharpness', 'ShearX',
                                              'ShearY', 'TranslateX', 'TranslateY',
                                              'Rotate']),
        transforms.ToTensor(),
        train_norm,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    pre_transform = transforms.Compose([
        transforms.RandomResizedCrop((config['h'], config['h']),
                                     scale=(0.5, 1.0), interpolation=InterpolationMode.BICUBIC),
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize',
                                              'Brightness', 'Sharpness', 'ShearX',
                                              'ShearY', 'TranslateX', 'TranslateY',
                                              'Rotate']),
        transforms.ToTensor(),
        train_norm,
        RandomErasing(probability=config['erasing_p'], mean=[0.0, 0.0, 0.0])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((config['h'], config['w']), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        test_norm,
    ])

    if dataset == 're_icfg':
        test_dataset = re_test_dataset_icfg(config, test_transform)
        if evaluate:
            return None, test_dataset
        train_dataset = re_train_dataset(config, train_transform, pre_transform)
        return train_dataset, test_dataset
    elif dataset == 're_pa100k':
        test_dataset = re_test_dataset_attr(config['test_file'], config, test_transform)
        val_dataset = re_test_dataset_attr(config['val_file'], config, test_transform)
        if evaluate:
            return None, val_dataset, test_dataset
        train_dataset = re_train_dataset_attr(config, train_transform)
        return train_dataset, val_dataset, test_dataset
    else:
        test_dataset = re_test_dataset(config['test_file'], config, test_transform)
        val_dataset = re_test_dataset(config['val_file'], config, test_transform)
        if evaluate:
            return None, val_dataset, test_dataset
        train_dataset = re_train_dataset(config, train_transform, pre_transform)
        return train_dataset, val_dataset, test_dataset


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset, shuffle in zip(datasets, shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank,
                                                      shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset, sampler, bs, n_worker, is_train, collate_fn in zip(datasets, samplers, batch_size, num_workers,
                                                                    is_trains, collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)

    if len(loaders) <= 1:
        print(f"### be careful: func create_loader returns a list length of {len(loaders)}")

    return loaders
