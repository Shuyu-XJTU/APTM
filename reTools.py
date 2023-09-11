import os
import math
import copy
import numpy as np
import time
import datetime
import json
from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image, ImageFont, ImageDraw
from sklearn import metrics
from easydict import EasyDict
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils


@torch.no_grad()
def evaluation_attr(model, data_loader, tokenizer, device, config, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation attr...')
    start_time = time.time()

    text = ['the person is a man', 'the person is a woman',
            'the person is no more than 60 years old', 'the person is older than 60 years old',
            'the person is a young or old one', 'the person is of mid age, between 18 and 60 years old',
            'the person is older than 18', 'the person is a baby or a teenager, younger than 18',

            'the picture is not the front of the person', 'the picture shows the front of the person',
            'the picture is not the side of the person', 'the picture shows the side of the person',
            'the picture is not the back of the person', 'the picture shows the back of the person',
            'a person without a hat', 'a person with a hat',

            'a person without a glasses', 'a person with a glasses',
            'a person without a handbag', 'a person with a handbag',
            'a person without a shoulder bag', 'a person with a shoulder bag',
            'a person without a backpack', 'a person with a backpack',

            'the person does not hold an object in front', 'the person hold an object in front',
            'the person does not wear short sleeved upper clothes', 'the person wears short sleeved upper clothes',
            'the person does not wear long sleeved upper clothes', 'the person wears long sleeved upper clothes',
            'there is no stride on the upper clothes of the person',
            'there is stride on the upper clothes of the person',

            'there is no logo on the upper clothes of the person',
            'there is logo on the upper clothes of the person',
            'there is no plaid on the upper clothes of the person',
            'there is plaid on the upper clothes of the person',
            'there is no splice on the upper clothes of the person',
            'there is splice on the upper clothes of the person',
            'there is no stripe on the upper clothes of the person',
            'there is stripe on the upper clothes of the person',

            'there is no pattern on the lower part of the person',
            'there is pattern on the lower part of the person',
            'the person does not wear long coat', 'the person wears long coat',
            'the person does not wear trousers', 'the person wears trousers',
            'the person does not wear shorts', 'the person wears shorts',

            'the person does not wear a skirt or a dress', 'the person wears a skirt or a dress',
            'the person does not wear boots', 'the person wears boots',
            ]

    text_input = tokenizer(text, padding='longest', max_length=config['max_tokens'],
                           return_tensors="pt").to(device)
    text_embeds = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
    text_atts = text_input.attention_mask

    image_embeds = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        image_embeds.append(image_embed)
    image_embeds = torch.cat(image_embeds, dim=0)

    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(text)), -1000.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = image_embeds.size(0) // num_tasks + 1
    start = rank * step
    end = min(image_embeds.size(0), start + step)

    for i, image_embed in enumerate(metric_logger.log_every(image_embeds[start:end], 50, header)):
        encoder_output = image_embed.repeat(len(text), 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.get_cross_embeds(encoder_output, encoder_att, text_embeds=text_embeds,
                                        text_atts=text_atts)[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_i2t[start + i] = score
    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_i2t, op=torch.distributed.ReduceOp.SUM)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))
    return score_matrix_i2t.cpu().numpy()


@torch.no_grad()
def evaluation_attr_only_img_classifier(model, data_loader, tokenizer, device, config, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'
    print('Computing features for evaluation attr...')
    start_time = time.time()

    image_embeds = []
    outputs = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed = model.vision_encoder(image)
        output = model.img_cls(image_embed[:, 0, :])
        output = torch.sigmoid(output)
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    orig_outputs = outputs.data.cpu().numpy()
    # transform raw outputs to attributes (binary codes)
    outputs = copy.deepcopy(orig_outputs)
    outputs[outputs < 0.5] = 0
    outputs[outputs >= 0.5] = 1
    return outputs


@torch.no_grad()
def accs(pred, y):
    print('Testing ... metrics')
    num_persons = pred.shape[0]
    print('num_persons', num_persons)
    ins_acc = 0
    ins_prec = 0
    ins_rec = 0
    mA_history = {
        'correct_pos': 0,
        'real_pos': 0,
        'correct_neg': 0,
        'real_neg': 0
    }

    # compute label-based metric
    outputs = pred
    attrs = y
    overlaps = outputs * attrs
    mA_history['correct_pos'] += overlaps.sum(0)
    mA_history['real_pos'] += attrs.sum(0)
    inv_overlaps = (1 - outputs) * (1 - attrs)
    mA_history['correct_neg'] += inv_overlaps.sum(0)
    mA_history['real_neg'] += (1 - attrs).sum(0)

    outputs = outputs.astype(bool)
    attrs = attrs.astype(bool)

    # compute instabce-based accuracy
    intersect = (outputs & attrs).astype(float)
    union = (outputs | attrs).astype(float)
    ins_acc += (intersect.sum(1) / union.sum(1)).sum()
    ins_prec += (intersect.sum(1) / outputs.astype(float).sum(1)).sum()
    ins_rec += (intersect.sum(1) / attrs.astype(float).sum(1)).sum()

    ins_acc /= num_persons
    ins_prec /= num_persons
    ins_rec /= num_persons
    ins_f1 = (2 * ins_prec * ins_rec) / (ins_prec + ins_rec)

    term1 = mA_history['correct_pos'] / mA_history['real_pos']
    term2 = mA_history['correct_neg'] / mA_history['real_neg']
    label_mA_verbose = (term1 + term2) * 0.5
    label_mA = label_mA_verbose.mean()

    print('* Results *')
    print('  # test persons: {}'.format(num_persons))
    print('  (label-based)     mean accuracy: {:.2%}'.format(label_mA))
    print('  (instance-based)  accuracy:      {:.2%}'.format(ins_acc))
    print('  (instance-based)  precition:     {:.2%}'.format(ins_prec))
    print('  (instance-based)  recall:        {:.2%}'.format(ins_rec))
    print('  (instance-based)  f1-score:      {:.2%}'.format(ins_f1))
    print('  mA for each attribute: {}'.format(label_mA_verbose))
    return label_mA, ins_acc, ins_prec, ins_rec, ins_f1


@torch.no_grad()
def itm_eval_attr(scores_i2t, dataset):
    label = dataset.label
    pred = []
    for i in range(label.shape[1]):
        a = np.argmax(scores_i2t[:, 2 * i: 2 * i + 2], axis=1)
        pred.append(a)

    label_mA, ins_acc, ins_prec, ins_rec, ins_f1 = accs(np.array(pred).T, label)
    print('############################################################\n')
    eval_result = {'label_mA': round(label_mA, 4),
                   'ins_acc': round(ins_acc, 4),
                   'ins_prec': round(ins_prec, 4),
                   'ins_rec': round(ins_rec, 4),
                   'ins_f1': round(ins_f1, 4),
                   }
    return eval_result


@torch.no_grad()
def itm_eval_attr_only_img_classifier(scores_i2t, dataset):
    label = dataset.label
    pred = scores_i2t
    label_mA, ins_acc, ins_prec, ins_rec, ins_f1 = accs(pred, label)
    print('############################################################\n')
    eval_result = {'label_mA': round(label_mA, 4),
                   'ins_acc': round(ins_acc, 4),
                   'ins_prec': round(ins_prec, 4),
                   'ins_rec': round(ins_rec, 4),
                   'ins_f1': round(ins_f1, 4),
                   }
    return eval_result


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'

    print('Computing features for evaluation...')
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']  # 256
    text_embeds = []
    text_atts = []
    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                               return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.text_proj(text_embed[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        text_embeds.append(text_embed)
        text_atts.append(text_input.attention_mask)
        text_feats.append(text_feat)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    image_embeds = []
    image_feats = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        image_feat = model.vision_proj(image_embed[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)
    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)
    sims_matrix = image_feats @ text_feats.t()
    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.image)),  1000.0).to(device)
    score_sim_t2i = sims_matrix

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx]
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)

        output = model.get_cross_embeds(encoder_output, encoder_att,
                                        text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1))[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score
        score_sim_t2i[start + i, topk_idx] = topk_sim

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002*score_sim_t2i

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    per_time = total_time / num_text
    print('total_time', total_time)
    print('per_time', per_time)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_t2i.cpu().numpy()


def mAP(scores_t2i, g_pids, q_pids, table=None):
    similarity = torch.tensor(scores_t2i)
    indices = torch.argsort(similarity, dim=1, descending=True)
    g_pids = torch.tensor(g_pids)
    q_pids = torch.tensor(q_pids)
    pred_labels = g_pids[indices.cpu()]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :10].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    t2i_cmc, t2i_mAP, t2i_mINP, _ = all_cmc, mAP, mINP, indices
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()

    if not table:
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        table.add_row(['t2i', t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP])
        table.custom_format["R1"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R5"] = lambda f, v: f"{v:.3f}"
        table.custom_format["R10"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mAP"] = lambda f, v: f"{v:.3f}"
        table.custom_format["mINP"] = lambda f, v: f"{v:.3f}"
        print(table)

    eval_result = {'R1': t2i_cmc[0],
                   'R5': t2i_cmc[4],
                   'R10': t2i_cmc[9],
                   'mAP': t2i_mAP,
                   'mINP': t2i_mINP,
                   }
    return eval_result
