import numpy as np
import json, codecs
from torch.utils.data import Dataset


from transformers import RobertaTokenizer
from torch.utils.data import DataLoader


def getTrainData(data_file, doc_file):

    pairs = []
    docs = {}
    docs_keys = []

    with codecs.open(doc_file, 'r', encoding='utf-8') as reader:
        context = reader.read()
        doc_dict = json.loads(context)

    with codecs.open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            info = line.strip().split('\t')
            # ['3086', 'What is the role of conversionism in Evangelicalism?', '672', 'Evangelicalism.html', '4']

            Q = info[1].strip()
            if Q not in docs:
                docs[Q] = []
                docs_keys.append(Q)
            doc_index = info[2]
            Answers = doc_dict[doc_index]
            len_Answers = len(Answers)
            Answer_index = [str(i) for i in range(len_Answers)]
            pos_index = info[-1].split(',')
            neg_index = sorted(list(set(Answer_index).difference(set(pos_index))), key=lambda x: int(x))
            for index in pos_index:
                A = Answers[index].strip()
                docs[Q].append(A)
                pairs.append([Q, A, 1])
            for index in neg_index:
                A = Answers[index].strip()
                docs[Q].append(A)
                pairs.append([Q, A, 0])
    return pairs, docs, docs_keys

def getAttackedData(attackedFile):
    attackedData = []
    with open(attackedFile, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            query, answer, label = line.strip().split('  [SEP]  ')
            label = int(label)
            attackedData.append([query, answer, label])

    return attackedData


class TrainData(Dataset):
    def __init__(self, data_file, doc_file, tokenizer, max_length, attacked_file=None):
        pairs, docs, docs_keys = getTrainData(data_file, doc_file)
        if attacked_file:
            attackedData = getAttackedData(attacked_file)
        else:
            attackedData = None
        self.tokenizer = tokenizer
        self.attackedData = attackedData
        self.pairs = pairs
        self.docs_keys = docs_keys
        self.docs = docs
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
