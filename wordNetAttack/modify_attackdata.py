#攻击后的数据格式与原本略有不同，将其与原本数据的格式统一

# import torch
# from torch.utils.data import Dataset, DataLoader
# from AttackDataset import AttackedData
import json
from tqdm import tqdm

attacked_file = r'D:\Dev\Data\WikiPassageQA\attack3\attacked_rate005_pos.json'
batch_size = 1


all_data = []
with open(attacked_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in tqdm(lines):
        new_data = {}
        a_data = json.loads(line)
        label = a_data['label']
        #修改input_ids  长度443
        input_ids = a_data['input_ids']
        sep_idx = input_ids.index(102)
        input_ids[:sep_idx] = input_ids[:44]
        if len(input_ids) > 443:
            input_ids = input_ids[:443]
            input_ids[-1] = 102
        elif len(input_ids) < 443:
            input_ids[:-1] += ([0 for i in range(443-len(input_ids))])

        #修改token_type_ids
        token_type_ids = [0 for i in range(45)] + [1 for j in range(45,443)]

        attention_mask = a_data['attention_mask']
        attention_mask = [1 for i in range(len(input_ids))]

        new_data['input_ids'] = input_ids
        new_data['token_type_ids'] = token_type_ids
        new_data['attention_mask'] = attention_mask
        new_data['label'] = label
        all_data.append(new_data)

with open('../data/new_attacked_data/new_rate005_pos.json', 'w', encoding='utf-8') as writer:
    for a_data in tqdm(all_data):
        json.dump(a_data, writer)
        writer.write('\n')

# with open('data/new_attacked_data/new_rate015_pos.json', 'r', encoding='utf-8') as reader:
#     lines = reader.readlines()
#     for line in tqdm(lines):
#         a_data = json.loads(line)
#         input_ids = a_data['input_ids']
#         token_type_ids = a_data['token_type_ids']
#         attention_mask = a_data['attention_mask']
#         if len(input_ids)!=443 or len(token_type_ids)!=443 or len(attention_mask)!=443:
#             print(a_data)

