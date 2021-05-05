#把分散攻击的train文件集成为1个

import csv

count = 0
all_datas = []
for i in range(1, 21):
    with open('data/sample_data/bae/train' + str(i) + '.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(float(row['ground_truth_output']))
            perturbed_text = row['perturbed_text']
            perturbed_text = perturbed_text.replace('Q1: ', '').replace('Q2: ', '').replace('>>>>', '\t') + '\t' + str(label)
            all_datas.append(perturbed_text)

with open('data/sample_data/bae/train_bae.txt', 'w', encoding='utf-8') as writer:
    for a_data in all_datas:
        writer.write(a_data.strip() + '\n')

# all_datas = []
# with open('data/sample_data/our_attack/train_attack.txt', 'r', encoding='utf-8') as reader:
#     lines = reader.readlines()
#     for line in lines:
#         a_data = line.strip().replace('  [SEP]  ', '\t')
#         all_datas.append(a_data)
#
#
# with open('data/sample_data/our_attack/train_ourattack.txt', 'w', encoding='utf-8') as writer:
#     for line in all_datas:
#         writer.write(line + '\n')