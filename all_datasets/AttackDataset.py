#attacked_datafile = 'data/test.json'
import json
import torch
import numpy as np

from model.MatchModel import  BertMatchModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


#model = BertMatchModel.from_pretrained('checkpoints/checkpoint-2')
#model.to('cuda')

def Dic2Tensor(dict):
    input_ids = np.array(dict['input_ids'])
    token_type_ids = np.array(dict['token_type_ids'])
    attention_mask = np.array(dict['attention_mask'])
    label = np.array(dict['label'])

    return input_ids, token_type_ids, attention_mask, label



class AttackedData(Dataset):
    def __init__(self, attacked_file):
        self.datas = []
        with open(attacked_file, 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            for line in lines:
                a_data = json.loads(line)
                self.datas.append(a_data)


    def __getitem__(self, index):
        a_data = self.datas[index]
        return Dic2Tensor(a_data)


    def __len__(self):
        # You should change 0 to the total size of your all_datasets.
        return len(self.datas)


# if __name__ == '__main__':
#     from torch.utils.data import DataLoader
#     attack_file = 'data/wikipassageQA/attaced_data2.json'
#     all_datasets = AttackedData(attacked_file=attack_file)
#     dataloader = DataLoader(all_datasets=all_datasets,
#                             batch_size=2,
#                             shuffle=False)
#     for batch in dataloader:
#         input_ids, token_type_ids, attention_mask, labels = batch
#         print(input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape)
#         break