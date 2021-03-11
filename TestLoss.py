from torch.utils.data import Dataset, DataLoader
from dataset import DataBert
from AttackDataset import AttackedData
from transformers import BertTokenizer,BertModel
import torch
from tqdm import tqdm
from model.MatchModel import BertMatchModel
from utils.FGM import FGM


train_file = 'data/wikipassageQA/train.tsv'
doc_file = 'data/wikipassageQA/document_passages.json'
attacked_file = 'data/attacked_data/new_rate1_pos.json'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertMatchModel.from_pretrained('checkpoints/checkpoint-2')
#model.to('cuda')

s1_length = 100
s2_length = 400
batch_size = 4
max_length = 0
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

attacked_data = AttackedData(attacked_file=attacked_file)  # 攻击样本

attack_dataloader = DataLoader(dataset=attacked_data,
                               batch_size=batch_size,
                               shuffle=False)
total_loss_clean = 0
total_loss_adv = 0
avg_loss_clean = 0
avg_loss_adv = 0
step = 0

model.train()
fgm = FGM(model)

count = 0
# with torch.no_grad():
for batch, batch_attack in tqdm(zip(train_dataLoader, attack_dataloader), desc="Iteration",
                                total=len(train_dataLoader)):
    # 设置tensor gpu运行
    model.zero_grad()
    step += 1
    #batch = tuple(t.to('cuda') for t in batch[:4])
    input_ids, token_type_ids, attention_mask, labels = batch[:4]

    outputs = model(input_ids=input_ids.long(),
                    token_type_ids=token_type_ids.long(),
                    labels=labels)

    loss_clean = outputs[0]
    loss_clean.backward()

    #print(loss_clean)
    total_loss_clean += loss_clean.cpu().item()
    avg_loss_clean = total_loss_clean / step


    #batch_attack = tuple(t.to('cuda') for t in batch_attack[:4])
    input_ids2, token_type_ids2, attention_mask2, labels2 = batch_attack[:4]

    fgm.attack()

    model.zero_grad()

    outputs_attack = model(input_ids=input_ids2.long(),
                           token_type_ids=token_type_ids2.long(),
                           attention_mask=attention_mask2,
                           labels=labels2)

    loss_adv = outputs_attack[0]
    total_loss_adv += loss_adv.cpu().item()
    avg_loss_adv = total_loss_adv / step

    fgm.restore()
    if loss_adv > loss_clean + 0.006:
        count += 1
    if step % 10 == 0:
        print(avg_loss_clean, avg_loss_adv, count)