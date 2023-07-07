import re
import json
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils
from tqdm import tqdm
from dataset.eda import *

def pre_caption(caption, max_words, icfg_rstp=False, is_eda=False, eda_p=0.5):
    if icfg_rstp:
        try:
            caption = re.sub(
                r'[^0-9a-z]+',
                ' ',
                caption.lower(),
            )
        except:
            print(caption)
        caption_words = caption.split()
        caption = ' '.join(caption_words)

    # eda
    if is_eda and random.random() < eda_p:
        caption = eda(caption, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=1)[0]

    # truncate caption
    caption_words = caption.split()
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    if not len(caption):
        raise ValueError("pre_caption yields invalid text")

    return caption
