#将原始数据转化为[Q, A ,label]

import json
pairs = []

def getBertData(data_file, doc_file, generate_file):
    doc_dict = None
    pairs = []

    with open(doc_file, 'r', encoding='utf-8') as reader:
        context = reader.read()
        doc_dict = json.loads(context)

    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            info = line.strip().split('\t')
            # ['3086', 'What is the role of conversionism in Evangelicalism?', '672', 'Evangelicalism.html', '4']

            Q = info[1].strip()
            doc_index = info[2]
            Answers = doc_dict[doc_index]
            len_Answers = len(Answers)
            Answer_index = [str(i) for i in range(len_Answers)]
            pos_index = info[-1].split(',')
            neg_index = sorted(list(set(Answer_index).difference(set(pos_index))), key=lambda x: int(x))
            for index in pos_index:
                pairs.append([Q, Answers[index].strip(), 1])
            for index in neg_index:
                pairs.append([Q, Answers[index].strip(), 0])
    return pairs
    # with open(generate_file, 'w', encoding='utf-8') as writer:
    #     for pair in pairs:
    #         writer.write(pair[0] + '\t' + pair[1] + '\t' + pair[0] + '\n\n')
    # print(len(pairs))

if __name__ == '__main__':
    data_file = 'data/wikipassageQA/train.tsv'
    doc_file = 'data/wikipassageQA/document_passages.json'
    generate_file = 'data/wikipassageQA/train_clean.txt'

    print(len(getBertData(data_file=data_file, doc_file=doc_file, generate_file=generate_file)))

