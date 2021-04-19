from transformers import BertTokenizer
import numpy as np

from torch.utils.data import Dataset, DataLoader



def getDatas(data_file):
    all_datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            query, answer, label = line.strip().split('\t')
            all_datas.append([query, answer, label])
    return all_datas
# data_file = '../data/sample_data/origin/train.txt'
# all_datas = getDatas(data_file=data_file)
# print(all_datas)

class SampleTrainData(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        pairs= getDatas(data_file)
        self.tokenizer = tokenizer
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
        return np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids']), np.array(tokenized_dic['attention_mask']), np.array([int(data[2])])

    def __len__(self):
        # You should change 0 to the total size of your all_datasets.
        return len(self.pairs)



# if __name__ == '__main__':
#     data_file = '../data/sample_data/origin/train.txt'
#     max_length = 400
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     batch_size = 2
#     shuffle = True
#     train_data = TrainData(data_file=data_file,
#                           max_length=max_length,
#                           tokenizer=tokenizer,
#                           )
#     train_dataLoader = DataLoader(dataset=train_data,
#                                 batch_size=batch_size,
#                                 shuffle= shuffle)
#
#
#     for batch in train_dataLoader:
#         print(batch)
#         break