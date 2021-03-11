#将数据切成多个碎片，方便并行进行攻击
data_file = '../data/wikipassageQA/train.tsv'

with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    print(len(lines))
    for i in range(1, 8):
        index = [(i-1)*476, i*476]
        slices = lines[index[0]:index[1]]
        write_file = f'data/wikipassageQA/splits/train/train_{str(i)}.tsv'
        with open(write_file, 'w', encoding='utf-8') as writer:
            for line in slices:
                writer.write(line)
