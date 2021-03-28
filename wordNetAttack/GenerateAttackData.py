#只替换了最重要的1个单词的替换文件（并没有使用，当时的测试想法而已）

#需要从命令行传入切割文件的数字号  python GenerateAttackData.py 1

from transformers import BertTokenizer
from model.MatchModel import BertMatchModel
from all_datasets.dataset import DataBert
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from utils import wordnettools as wntool
import json

checkpoints = 'checkpoints/checkpoint-2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(checkpoints) #'bert-base-uncased'
model = BertMatchModel.from_pretrained(checkpoints)
print('device:', device)

model.to(device)

#file_number = sys.argv[1]
#train_file = f'data/wikipassageQA/splits/train/train_{file_number}.tsv'

train_file = '../data/wikipassageQA/train2.tsv'
doc_file = '../data/wikipassageQA/document_passages.json'

s1_length = 100
s2_length = 400
max_length = 0
batch_size = 8

train_data = DataBert(data_file=train_file,
                      doc_file=doc_file,
                      s1_length=s1_length,
                      s2_length=s2_length,
                      max_length=max_length,
                      tokenizer=tokenizer
                      )

train_dataLoader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              shuffle=False)


special_tokens = [100, 101, 102, 103]
#mask为103
max_query = 0
max_answer = 0
query = ''
answer = ''
attcked_data = []
print(train_data.__len__())
with torch.no_grad():
    for batch in tqdm(train_dataLoader, desc="Iteration"):
        # 设置tensor gpu运行
        input_ids, token_type_ids, attention_mask, labels, origin_query, origin_answers = batch
        inputs = (input_ids, token_type_ids, attention_mask, labels)
        inputs_device = tuple(t.to(device) for t in inputs)
        input_ids, token_type_ids, attention_mask, labels = inputs_device


        for input_id, token_type_id, attention, label in zip(input_ids.long(), token_type_ids.long(), attention_mask, labels):  #遍历batch的每一句话
            for k in range(len(input_id)):  #将pad部分attention置0
                if input_id[k].item() == 0:
                    attention[k] = 0
            #print(attention)
            outputs = model(input_ids=input_id.unsqueeze(0), token_type_ids=token_type_id, attention_mask=attention.unsqueeze(0), labels=label.unsqueeze(0))
            origin_score = outputs[0]
            #print('origin_loss:', origin_score)
            all_inputs_for_a_data = []
            for i in range(len(input_id)):
                if input_id[i].item() in special_tokens:  #查看该token是否在特殊tokens里
                    continue
                elif input_id[i].item() == 0:
                    continue
                else:
                    new_input = {}  # 用字典来装每一条数据 'input_ids', 'token_type_ids', 'attention_mask', 'label', 'position', 'origin_id', 'loss'
                    new_input['origin_loss'] = origin_score.item()
                    new_input_id, new_attention, new_label = input_id.clone().detach(), attention.clone().detach(), label.clone().detach()
                    origin_id = new_input_id[i].item()
                    new_attention[i] = 0
                    #print(new_attention)
                    new_input_id = new_input_id.unsqueeze(0)
                    new_attention = new_attention.unsqueeze(0)
                    new_label = new_label.unsqueeze(0)
                    new_input['input_ids'] = new_input_id
                    new_input['token_type_ids'] = token_type_id
                    new_input['attention_mask'] = new_attention
                    new_input['label'] = new_label
                    new_input['position'] = i
                    new_input['origin_id'] = origin_id

                    outputs = model(input_ids=new_input_id, token_type_ids=token_type_id, attention_mask=new_attention, labels=new_label)
                    loss = outputs[0]
                    new_input['loss'] = loss.item()
                    all_inputs_for_a_data.append(new_input)  #将一条数据的所有替换结果记录到list中

            #print(len(all_inputs_for_a_data))
            all_inputs_for_a_data.sort(key=lambda x:x['loss'], reverse=True)  #按照loss大小从大到小排序

            top3 = all_inputs_for_a_data[:3]

            flag = 0
            for dic in top3:
                #print(dic['loss'], dic['position'], tokenizer.convert_ids_to_tokens(dic['origin_id']))
                origin_word = tokenizer.convert_ids_to_tokens(dic['origin_id'])
                backup_words = wntool.GenerateSimWordByWN(origin_word)
                if len(backup_words) == 0:
                    continue
                else:
                    for word in backup_words:
                        word_id = tokenizer.convert_tokens_to_ids(word)
                        if word_id == 100:
                            continue
                        else:
                            dic['replace_id'] = word_id
                            position= dic['position']
                            ids = dic['input_ids']
                            ids[0][position] = word_id
                            dic['input_ids'] = ids
                            atten = dic['attention_mask']
                            atten[0][position] = 1
                            dic['attention_mask'] = atten
                            attcked_data.append(dic)
                            flag = 1   #如果完成了替换，就将flag=1，退出整个数据的循环
                            break
                    if flag == 1:
                        break  #替换完成跳出循环
            else:
                position= dic['position']    #如果替换失败，就将原来的句子作为结果
                atten = dic['attention_mask']
                atten[0][position] = 1
                attcked_data.append(dic)

            #break  #完成一条数据处理
        #break
        #跳出一个batch
    #print(attcked_data)
#with open(f'data/wikipassageQA/splits/attack/attacked_train_{file_number}.json', 'w') as writer:
with open(f'data/wikipassageQA/attacked_train.json', 'w') as writer:
    for dict in tqdm(attcked_data, desc='writing'):
        for item in dict.items():
            if isinstance(item[1],torch.Tensor):
                dict[item[0]] = item[1].cpu().numpy().tolist()
        json.dump(dict, writer)
        writer.write('\n')





