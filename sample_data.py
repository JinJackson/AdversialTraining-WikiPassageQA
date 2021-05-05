import random

with open('data/wikipassageQA/text/train.txt', 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
with open('data/AttackedText/rate1_pos_text.txt', 'r', encoding='utf-8') as reader:
    lines_attack = reader.readlines()

length = len(lines)

nums_list = list(range(length))

shuffle_selected = random.sample(nums_list, 40000)

with open('data/sample_data/origin/train.txt', 'w', encoding='utf-8') as writer:
    for index in shuffle_selected:
        writer.write(lines[index].strip() + '\n')


with open('data/sample_data/our_attack/train_attack.txt', 'w', encoding='utf-8') as writer:
    for index in shuffle_selected:
        writer.write(lines_attack[index].strip() + '\n')
