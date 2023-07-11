import argparse
import os
import math

import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from prettytable import PrettyTable

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_retrieval import APTM_Retrieval
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader
from dataset.re_dataset import TextMaskingGenerator
from scheduler import create_scheduler
from optim import create_optimizer

from trains import train, train_attr
from train_pa100ks import train_pa100k, train_pa100k_only_img_classifier

from reTools import evaluation, mAP
from reTools import evaluation_attr, itm_eval_attr
from reTools import evaluation_attr_only_img_classifier, itm_eval_attr_only_img_classifier


def main(args, config):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    world_size = utils.get_world_size()

    if args.bs > 0:
        config['batch_size_train'] = args.bs // world_size
    if args.epo > 0:
        config['schedular']['epochs'] = args.epo

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    print("Creating model", flush=True)
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = APTM_Retrieval(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint, config, is_eval=args.evaluate)
    model = model.to(device)

    print("### Total Params: ", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Creating retrieval dataset", flush=True)
    if args.task == "itr_icfg":
        train_dataset, test_dataset = create_dataset('re_icfg', config, args.evaluate)
    elif args.task == "itr_rstp":
        train_dataset, val_dataset, test_dataset = create_dataset('re_rstp', config, args.evaluate)
    elif args.task == "itr_cuhk":
        train_dataset, val_dataset, test_dataset = create_dataset('re_cuhk', config, args.evaluate)
    elif args.task == "itr_pa100k":
        train_dataset, val_dataset, test_dataset = create_dataset('re_pa100k', config, args.evaluate)
    else:
        train_dataset, val_dataset, test_dataset = create_dataset('re_gene', config, args.evaluate)

    start_time = time.time()
    print("### output_dir, ", args.output_dir, flush=True)

    if args.evaluate:
        print("Start evaluating", flush=True)
        # if args.task not in ["itr_icfg", "itr_pa100k"]:
        #     print("val_dataset", flush=True)
        #     val_loader = create_loader([val_dataset], [None],
        #                                batch_size=[config['batch_size_test']],
        #                                num_workers=[4],
        #                                is_trains=[False],
        #                                collate_fns=[None])[0]
        #     score_val_t2i = evaluation(model_without_ddp, val_loader,
        #                                                tokenizer, device, config, args)

        print("test_dataset", flush=True)
        test_loader = create_loader([test_dataset], [None],
                                    batch_size=[config['batch_size_test']],
                                    num_workers=[4],
                                    is_trains=[False],
                                    collate_fns=[None])[0]
        if args.task == "itr_pa100k":
            if model_without_ddp.pa100k_only_img_classifier:
                score_test_i2t_attr = evaluation_attr_only_img_classifier(model_without_ddp, test_loader,
                                                                          tokenizer, device, config, args)
            else:
                score_test_i2t_attr = evaluation_attr(model_without_ddp, test_loader,
                                                      tokenizer, device, config, args)
        else:
            score_test_t2i = evaluation(model_without_ddp, test_loader,
                                                        tokenizer, device, config, args)

        if utils.is_main_process():
            # if args.task not in ["itr_icfg", "itr_pa100k"]:
            #     print('val_result:', flush=True)
            #     mAP(score_val_t2i, val_loader.dataset.g_pids, val_loader.dataset.q_pids)
            if args.task == "itr_pa100k":
                if model_without_ddp.pa100k_only_img_classifier:
                    test_result_attr = itm_eval_attr_only_img_classifier(score_test_i2t_attr, test_loader.dataset)
                else:
                    test_result_attr = itm_eval_attr(score_test_i2t_attr, test_loader.dataset)
                print('test_result_attr:', flush=True)
                print(test_result_attr, flush=True)
            else:
                print('test_result:', flush=True)
                mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids)

        dist.barrier()

    else:
        print("Start training", flush=True)
        train_dataset_size = len(train_dataset)
        if utils.is_main_process():
            print(f"### data {train_dataset_size}, batch size, {config['batch_size_train']} x {world_size}")
            if args.task == "itr_pa100k":
                table = PrettyTable(["epoch", "label_mA", "ins_acc", "ins_prec", "ins_rec", "ins_f1"])
            else:
                table = PrettyTable(["epoch", "R1", "R5", "R10", "mAP", "mINP"])
                table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
                table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
                table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
                table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
                table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            if args.task == "itr_icfg":
                samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None]
            else:
                samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
        else:
            if args.task == "itr_icfg":
                samplers = [None, None]
            else:
                samplers = [None, None, None]

        if args.task == "itr_icfg":
            train_loader, test_loader = create_loader([train_dataset, test_dataset], samplers,
                                                      batch_size=[config['batch_size_train']] + [
                                                          config['batch_size_test']],
                                                      num_workers=[4, 4], is_trains=[True, False],
                                                      collate_fns=[None, None])
        else:
            train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset], samplers,
                                                                  batch_size=[config['batch_size_train']] + [
                                                                      config['batch_size_test']] * 2,
                                                                  num_workers=[4, 4, 4], is_trains=[True, False, False],
                                                                  collate_fns=[None, None, None])

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model_without_ddp)
        arg_sche = utils.AttrDict(config['schedular'])
        arg_sche['step_per_epoch'] = math.ceil(train_dataset_size / (config['batch_size_train'] * world_size))
        lr_scheduler = create_scheduler(arg_sche, optimizer)

        max_epoch = config['schedular']['epochs']
        best = 0
        best_epoch = 0

        if config['mlm']:
            mask_generator = TextMaskingGenerator(tokenizer, config['mask_prob'], config['max_masks'],
                                                  config['skipgram_prb'], config['skipgram_size'],
                                                  config['mask_whole_word'])
        else:
            mask_generator = None

        for epoch in range(0, max_epoch):
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            if args.task == "itr_pa100k":
                if model_without_ddp.pa100k_only_img_classifier:
                    train_stats = train_pa100k_only_img_classifier(model, train_loader, optimizer, tokenizer, epoch,
                                                                   device, lr_scheduler, config, mask_generator)
                else:
                    train_stats = train_pa100k(model, train_loader, optimizer, tokenizer, epoch,
                                               device, lr_scheduler, config, mask_generator)
            else:
                if ('attr' in config.keys()) and config['attr']:
                    train_stats = train_attr(model, train_loader, optimizer, tokenizer, epoch,
                                             device, lr_scheduler, config, mask_generator)
                else:
                    train_stats = train(model, train_loader, optimizer, tokenizer, epoch,
                                        device, lr_scheduler, config, mask_generator)

            if (epoch + 1) % 1 == 0:
                # if args.task not in ["itr_icfg", "itr_pa100k"]:
                #     score_val_t2i = evaluation(model_without_ddp, val_loader, tokenizer,
                #                                               device, config, args)
                if args.task == "itr_pa100k":
                    if model_without_ddp.pa100k_only_img_classifier:
                        score_test_i2t = evaluation_attr_only_img_classifier(model_without_ddp, test_loader,
                                                                             tokenizer, device, config, args)
                    else:
                        score_test_i2t = evaluation_attr(model_without_ddp, test_loader,
                                                         tokenizer, device, config, args)
                else:
                    score_test_t2i = evaluation(model_without_ddp, test_loader,
                                                                tokenizer, device, config, args)

            if utils.is_main_process():
                # if args.task not in ["itr_icfg", "itr_pa100k"]:
                #     val_result = mAP(score_val_t2i, val_loader.dataset.g_pids, val_loader.dataset.q_pids, table)
                if args.task == "itr_pa100k":
                    if model_without_ddp.pa100k_only_img_classifier:
                        test_result = itm_eval_attr_only_img_classifier(score_test_i2t, test_loader.dataset)
                    else:
                        test_result = itm_eval_attr(score_test_i2t, test_loader.dataset)
                    table.add_row([epoch, test_result['label_mA'] * 100, test_result['ins_acc'] * 100,
                                   test_result['ins_prec'] * 100, test_result['ins_rec'] * 100,
                                   test_result['ins_f1'] * 100])
                    test_result_log = test_result
                else:
                    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                    table.add_row([epoch, test_result['R1'], test_result['R5'], test_result['R10'],
                                   test_result['mAP'], test_result['mINP']])
                    test_result_log = {}
                    for k, v in test_result.items():
                        test_result_log[k] = str(np.around(v, 3))
                print(table, flush=True)

                log_stats = {'e': epoch,
                             **{k: v for k, v in test_result_log.items()},
                             **{k: v for k, v in train_stats.items()},
                             # **{f'val_{k}': v for k, v in val_result.items()},
                             }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                if args.task == "itr_pa100k":
                    result = test_result['label_mA']
                else:
                    result = test_result['R1']

                if result > best:
                    save_obj = {'model': model_without_ddp.state_dict(), 'config': config, }
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                    best = result
                    best_epoch = epoch
                elif epoch >= max_epoch - 1:
                    save_obj = {
                        'model': model_without_ddp.state_dict(),
                        # 'optimizer': optimizer.state_dict(),
                        # 'lr_scheduler': lr_scheduler.state_dict(),
                        'config': config,
                        # 'epoch': epoch,
                    }
                    torch.save(save_obj, os.path.join(args.output_dir, f'checkpoint_{epoch}.pth'))
            dist.barrier()
            torch.cuda.empty_cache()

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write("best epoch: %d" % best_epoch)

            os.system(f"cat {args.output_dir}/log.txt")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(' ### Time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument('--bs', default=-1, type=int, help="for each gpu, batch_size = bs // num_gpus")
    parser.add_argument('--epo', default=-1, type=int, help="epoch")
    parser.add_argument('--evaluate', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
