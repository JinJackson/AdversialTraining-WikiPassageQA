#测试攻击后的数据数量与原本数量是否相同

from transformers import BertTokenizer
origin_file = r'D:\Dev\Data\WikiPassageQA\train\train_.tsv'
attack_file = r'D:\Dev\Data\WikiPassageQA\attack'
doc_file = 'data/wikipassageQA/document_passages.json'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

from all_datasets.dataset import DataBert


ori_lines = []
attack_lines = []

for i in range(4, 6):

    train_file = rf'D:\Dev\pythonspace\pycharmWorkspace\Experiments\FGM-WordNet\wkpqa\data\wikipassageQA\splits\train\train_{str(i)}.tsv'
    datasets = DataBert(data_file=train_file,
                     doc_file=doc_file,
                     s1_length=100,
                     s2_length=400,
                     max_length=0,
                     tokenizer=tokenizer)
    ori_lines.append(len(datasets))



    with open(rf'D:\Dev\pythonspace\pycharmWorkspace\Experiments\FGM-WordNet\wkpqa\data\wikipassageQA\splits2\attacked_train_{str(i)}.json', 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        attack_lines.append(len(lines))

print(ori_lines)
print(attack_lines)
