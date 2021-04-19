data_file = 'data/sample_data/origin/train.txt'
with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()

length = len(lines)

splits = 10
assert length % splits == 0

file_size = int(length / splits)

count = 0
for i in range(1, splits+1):
    written_file = 'data/sample_data/origin/train' + str(i) + '.txt'
    with open(written_file, 'w', encoding='utf-8') as writer:
        start = count
        print(start)
        for j in range(start, i*file_size):
            writer.write(lines[j].strip() + '\n')
            count += 1