import numpy as np
import json, codecs
from torch.utils.data import Dataset


from transformers import RobertaTokenizer
from torch.utils.data import DataLoader


def getOriginDatas(data_file):
    all_datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            query, answer, label = line.strip().split('\t')
            label = int(label)
            all_datas.append([query, answer, label])
    return all_datas

def getAttackedData(attackedFile):
    attackedData = []
    with open(attackedFile, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            query, answer, label = line.strip().split('\t')
            label = int(label)
            attackedData.append([query, answer, label])

    return attackedData

#这个dataset类只用作训练，BertDataset中的TrainData可以用来加载dev和test，这个dataset没有记录docs之类的，无法用于dev和test
class AttackTrainData(Dataset):
    def __init__(self, data_file, tokenizer, max_length, attacked_file=None):
        pairs = getOriginDatas(data_file)
        if attacked_file:
            attackedData = getAttackedData(attacked_file)
        else:
            attackedData = None
        self.tokenizer = tokenizer
        self.attackedData = attackedData
        self.pairs = pairs
        self.max_length = max_length
    def __getitem__(self, index):
        data = self.pairs[index]
        query, answer, label = data
        tokenized_dic = self.tokenizer.encode_plus(text=query,
                                          text_pair=answer,
                                          truncation=True,
                                          padding='max_length',
                                          max_length=self.max_length)
        if self.attackedData:
            adv_data = self.attackedData[index]
            adv_query, adv_answer, adv_label = adv_data
            adv_tokenized_dic = self.tokenizer.encode_plus(text=adv_query,
                                                           text_pair=adv_answer,
                                                           truncation=True,
                                                           padding='max_length',
                                                           max_length=self.max_length)

            return [np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids']), np.array(tokenized_dic['attention_mask']), np.array([data[2]])], \
                   [np.array(adv_tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids']), np.array(adv_tokenized_dic['attention_mask']), np.array([adv_data[2]])]

        return [np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids']), np.array(tokenized_dic['attention_mask']), np.array([data[2]])]

    def __len__(self):
        # You should change 0 to the total size of your all_datasets.
        return len(self.pairs)

# if __name__ == '__main__':
#     from transformers import BertTokenizer
#     train_file = '../data/sample_data/origin/train.txt'
#     attack_file = '../data/sample_data/bae/train_bae.txt'
#     max_length = 20
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     batch_size = 2
#     shuffle = True
#     train_data = AttackTrainData(data_file=train_file,
#                           max_length=max_length,
#                           tokenizer=tokenizer,
#                           attacked_file=attack_file
#                           )
#     train_dataLoader = DataLoader(dataset=train_data,
#                                 batch_size=batch_size,
#                                 shuffle= shuffle)
#
#
#     for batch in train_dataLoader:
#         print(batch)
#         break