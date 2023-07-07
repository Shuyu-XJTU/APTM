import json
import os
import random

import numpy as np
from random import randint, shuffle
from random import random as rand
from PIL import Image
from PIL import ImageFile

import torch
from torch.utils.data import Dataset

from dataset.utils import pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class TextMaskingGenerator:
    def __init__(self, tokenizer, mask_prob, mask_max, skipgram_prb=0.2, skipgram_size=3, mask_whole_word=True,
                 use_roberta=False):
        self.id2token = {i: w for w, i in tokenizer.get_vocab().items()}
        self.use_roberta = use_roberta
        for i in range(len(self.id2token)):
            assert i in self.id2token.keys()  # check
        self.cls_token_id = tokenizer.cls_token_id
        self.mask_token_id = tokenizer.mask_token_id
        self.mask_max = mask_max
        self.mask_prob = mask_prob
        self.skipgram_prb = skipgram_prb
        self.skipgram_size = skipgram_size
        self.mask_whole_word = mask_whole_word

        print("len(tokenizer.id2token): ", len(self.id2token), "  ----  cls_token_id: ", self.cls_token_id,
              "  ----  mask_token_id: ", self.mask_token_id, flush=True)

    def get_random_word(self):
        i = randint(0, len(self.id2token) - 1)
        return i  # self.id2token[i]

    def __call__(self, text_ids):  # tokens: [CLS] + ...
        n_pred = min(self.mask_max, max(1, int(round(len(text_ids) * self.mask_prob))))

        # candidate positions of masked tokens
        assert text_ids[0] == self.cls_token_id
        special_pos = set([0])  # will not be masked
        cand_pos = list(range(1, len(text_ids)))

        shuffle(cand_pos)
        masked_pos = set()
        max_cand_pos = max(cand_pos)
        for pos in cand_pos:
            if len(masked_pos) >= n_pred:
                break
            if pos in masked_pos:
                continue

            def _expand_whole_word(st, end):
                new_st, new_end = st, end

                if self.use_roberta:
                    while (new_st > 1) and (self.id2token[text_ids[new_st].item()][0] != 'Ġ'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and (self.id2token[text_ids[new_end].item()][0] != 'Ġ'):
                        new_end += 1
                else:
                    # bert, WordPiece
                    while (new_st >= 0) and self.id2token[text_ids[new_st].item()].startswith('##'):
                        new_st -= 1
                    while (new_end < len(text_ids)) and self.id2token[text_ids[new_end].item()].startswith('##'):
                        new_end += 1

                return new_st, new_end

            if (self.skipgram_prb > 0) and (self.skipgram_size >= 2) and (rand() < self.skipgram_prb):
                # ngram
                cur_skipgram_size = randint(2, self.skipgram_size)
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(
                        pos, pos + cur_skipgram_size)
                else:
                    st_pos, end_pos = pos, pos + cur_skipgram_size
            else:
                if self.mask_whole_word:
                    st_pos, end_pos = _expand_whole_word(pos, pos + 1)
                else:
                    st_pos, end_pos = pos, pos + 1

            for mp in range(st_pos, end_pos):
                if (0 < mp <= max_cand_pos) and (mp not in special_pos):
                    masked_pos.add(mp)
                else:
                    break

        masked_pos = list(masked_pos)
        n_real_pred = len(masked_pos)
        if n_real_pred > n_pred:
            shuffle(masked_pos)
            masked_pos = masked_pos[:n_pred]

        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                text_ids[pos] = self.mask_token_id
            elif rand() < 0.5:  # 10%
                text_ids[pos] = self.get_random_word()

        return text_ids, masked_pos


class re_train_dataset(Dataset):
    def __init__(self, config, transform, pre_transform):
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']
        self.eda = config['eda']
        self.eda_p = config['eda_p']
        ann_file = config['train_file']

        if ('attr' in config.keys()) and config['attr']:
            self.attr = True
        else:
            self.attr = False

        self.transform = transform
        self.pre_transform = pre_transform
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))

        self.img_ids = {}

        n = 1
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]
        try:
            image_path = os.path.join(self.image_root, ann['image'])
        except:
            print("self.image_root", self.image_root)
            print("ann['image']", ann['image'])
        image = Image.open(image_path).convert('RGB')
        image1 = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)
        if self.eda:
            caption1 = pre_caption(ann['caption'], self.max_words, self.icfg_rstp, True, self.eda_p)
            return image1, caption, caption1, self.img_ids[ann['image_id']]
        elif self.attr:
            label = torch.tensor(ann['label'])
            return image1, caption, self.img_ids[ann['image_id']], label
        else:
            return image1, caption, self.img_ids[ann['image_id']]


class re_test_dataset(Dataset):
    def __init__(self, ann_file, config, transform):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']
        self.icfg_rstp = config['icfg_rstp']

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.g_pids = []
        self.q_pids = []

        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.g_pids.append(ann['image_id'])
            self.image.append(ann['image'])
            self.img2txt[img_id] = []

            t = 0
            for i, caption in enumerate(ann['caption']):
                self.q_pids.append(ann['image_id'])
                self.text.append(pre_caption(caption, self.max_words, icfg_rstp=self.icfg_rstp))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = []
                self.txt2img[txt_id].append(img_id)
                txt_id += 1
                t += 1

            txt_id1 = 0
            for img_id1, ann1 in enumerate(self.ann):
                for i1, caption1 in enumerate(ann1['caption']):
                    if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                        self.img2txt[img_id].append(txt_id1)
                    txt_id1 += 1
                if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                    for temp in range(t):
                        self.txt2img[txt_id - 1 - temp].append(img_id1)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class re_test_dataset_icfg(Dataset):
    def __init__(self, config, transform):
        ann_file = config['test_file']
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        self.g_pids = []
        self.q_pids = []

        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.g_pids.append(ann['image_id'])
            self.img2txt[img_id] = []
            self.img2txt[img_id].append(img_id)

            self.text.append(pre_caption(ann['caption'][0], self.max_words, icfg_rstp=True))
            self.q_pids.append(ann['image_id'])

            self.txt2img[img_id] = []
            self.txt2img[img_id].append(img_id)

            for img_id1, ann1 in enumerate(self.ann):
                if ann['image_id'] == ann1['image_id'] and img_id != img_id1:
                    self.txt2img[img_id].append(img_id1)
                    self.img2txt[img_id].append(img_id1)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index


class re_train_dataset_attr(Dataset):
    def __init__(self, config, transform):
        ann_file = config['train_file']
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        label = torch.tensor(ann['label'])
        return image, label


class re_test_dataset_attr(Dataset):
    def __init__(self, ann_file, config, transform):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.image_root = config['image_root']
        self.max_words = config['max_words']

        self.image = []
        self.label = []
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.label.append(ann['label'])
        self.label = np.array(self.label)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index
