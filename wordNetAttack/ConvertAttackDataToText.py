import json
from transformers import BertTokenizer
from tqdm import tqdm
Bert_Attack_File = '../data/attacked_data/new_rate1_pos.json'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
written_file = '../data/AttackedDataTextForm/rate1_pos_text.txt'

def clearSpecialTokens(sent):
    sent = sent.replace('[CLS]', '').replace('[PAD]', '').strip()
    text_pairs = sent.split('[SEP]')
    text_pairs = [text.strip() for text in text_pairs if text]
    return text_pairs

def convertData(data_file):

    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        print('len of whole data:', len(lines))
        for line in tqdm(lines, desc='loading'):
            a_data = json.loads(line)
            input_ids = a_data['input_ids']  #取出ids
            label = str(a_data['label'][0])
            tokens = tokenizer.convert_ids_to_tokens(input_ids)  #转成token
            whole_sent = tokenizer.convert_tokens_to_string(tokens)  #变成string

            text_pairs = clearSpecialTokens(whole_sent)  #把特殊标记清除， 按照[SEP]分开
            text_pairs.append(label)
            datas.append(text_pairs)
    return datas

def writeData(data, written_file):
    with open(written_file, 'w', encoding='utf-8') as writer:
        for text_pair in tqdm(data, desc='writing'):
            string_text = '  [SEP]  '.join(text_pair)
            writer.write(string_text + '\n')






    #print(count)
            # break


if __name__ == '__main__':
    datas = convertData(data_file=Bert_Attack_File)
    writeData(datas, written_file=written_file)
