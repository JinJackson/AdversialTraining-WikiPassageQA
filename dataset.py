import numpy as np
import json, codecs
from torch.utils.data import Dataset

def pad_sent(sent, pad, max_length):
    sent_arr = sent.split()
    sent_arr.extend([pad for i in range(max_length)])
    return ' '.join(sent_arr[:max_length]).strip()

def getBertData(data_file, doc_file):

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

class DataBert(Dataset):
    def __init__(self, data_file, doc_file, s1_length, s2_length, max_length, tokenizer):
        pairs, docs, docs_keys = getBertData(data_file, doc_file)

        self.s1_length = s1_length
        self.s2_length = s2_length
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.pairs = pairs
        self.docs_keys = docs_keys
        self.docs = docs
    def __getitem__(self, index):
        data = self.pairs[index]
        if self.max_length == 0:
          query = pad_sent(data[0], '[PAD]', self.s1_length)
          answer = pad_sent(data[1], '[PAD]', self.s2_length)
          max_length = self.s1_length + self.s2_length + 3
        else:
          query = data[0]
          answer = data[1]
          max_length = self.max_length

        tokenized_dic = self.tokenizer.encode_plus(text=query,
                                          text_pair=answer,
                                          max_length=max_length,
                                          truncation=True,
                                          padding='max_length')

        return np.array(tokenized_dic['input_ids']), np.array(tokenized_dic['token_type_ids']), np.array(tokenized_dic['attention_mask']), np.array([data[2]]), query, answer

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.pairs)



# if __name__ == '__main__':
#     from transformers import BertTokenizer
#     from torch.utils.data import DataLoader
#     train_file = 'data/wikipassageQA/train.tsv'
#     doc_file = 'data/wikipassageQA/document_passages.json'
#     s1_length = 100
#     s2_length = 400
#     max_length = 0
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#     train_data = DataBert(data_file=train_file,
#                           doc_file=doc_file,
#                           s1_length=s1_length,
#                           s2_length=s2_length,
#                           max_length=max_length,
#                           tokenizer=tokenizer
#                           )
#
#     dataloader = DataLoader(dataset=train_data,
#                             batch_size=2,
#                             shuffle=False)
#
#     for batch in dataloader:
#         input_ids, token_type_ids, attention_mask, labels = batch[:4]
#         print(input_ids.shape, token_type_ids.shape, attention_mask.shape, labels.shape)
#         break

