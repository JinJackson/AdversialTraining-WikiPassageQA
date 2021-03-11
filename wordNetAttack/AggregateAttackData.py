#将切成多个片的训练数据在攻击后重新拼回一个数据
import json
from tqdm import tqdm
count = 0
with open('data/wikipassageQA/attack_data_allwords.json', 'w', encoding='utf-8') as writer:
    for i in tqdm(range(1, 8)):
        with open(f'data/wikipassageQA/splits2/attacked_train_{str(i)}.json', 'r', encoding='utf-8') as reader:
            lines = reader.readlines()
            for line in lines:
                count += 1
                a_data = json.loads(line)
                json.dump(a_data ,writer)
                writer.write('\n')

print(count)
