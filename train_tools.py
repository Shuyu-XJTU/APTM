import torch
import math


def mlm(text, text_input, tokenizer, device, mask_generator, config, pa100k=False):
    if pa100k:
        text_masked = tokenizer(text, padding='longest', max_length=config['max_tokens'],
                                return_tensors="pt").to(device)
    else:
        text_masked = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                return_tensors="pt").to(device)
    text_ids_masked = text_masked.input_ids
    masked_pos = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.int64, device=device)
    masked_ids = torch.empty((text_ids_masked.shape[0], config['max_masks']), dtype=torch.long, device=device)
    for index, text_id in enumerate(text_ids_masked):
        text_ids_masked_, masked_pos_ = mask_generator(text_id)
        masked_ids_ = [text_input.input_ids[index][p].item() for p in masked_pos_]
        n_pad = config['max_masks'] - len(masked_ids_)
        masked_pos_ = masked_pos_ + [0] * n_pad
        masked_pos_ = torch.tensor(masked_pos_, dtype=torch.int64).to(device)
        masked_ids_ = masked_ids_ + [-100] * n_pad
        masked_ids_ = torch.tensor(masked_ids_, dtype=torch.long).to(device)
        masked_pos[index] = masked_pos_
        masked_ids[index] = masked_ids_
    return text_ids_masked, masked_pos, masked_ids
