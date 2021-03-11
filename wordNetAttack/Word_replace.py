import json
import nltk
import pdb
from sklearn.metrics.pairwise import cosine_similarity
import time
from tqdm import tqdm
dict_file = '../data/wordVectors/word_dict.json'  #词向量json文件
# attack_data_file = 'data/wikipassageQA/sample_data.json'
attack_data_file = 'data/wikipassageQA/attack_data_allwords.json'
from transformers import BertTokenizer
from nltk.corpus import stopwords
from utils.wordnettools import WNtools
#nltk.download('stopwords')

wntools = WNtools()

# 返回一个dict，key为单词，value为其embedding
# 暂时人为判断替换词应该在0.6以上
def CounterFittedVectors(dict_file):
    all_words = {}
    with open(dict_file, 'r', encoding='utf-8') as reader:
        all_data = reader.read()
        word_dict = json.loads(all_data)
        for dict in word_dict:
            all_words.update(dict)
    return all_words


#读入数据
def getAttackData(attack_data_file):
    all_data = []
    with open(attack_data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            a_data = json.loads(line)
            all_data.append(a_data)

    return all_data

#检查是否为 后缀 or 标点符号 or 停用词
def checkwords(aword, stopwords):
    punctuate = '[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。?、~@#￥%……&*（）]+'
    if ('#' in aword) or (aword in punctuate) or (aword in stopwords):
        return False
    else:
        return True

#计算两个词在counter-fitted embedding中的余弦相似度
def cal_cos_sim(word1, word2, word_dict):
    embedding1 = word_dict.get(word1, None)
    embedding2 = word_dict.get(word2, None)
    if embedding1 and embedding2:
        results = cosine_similarity([embedding1, embedding2])[0][1]
        return results
    else:
        return None

def attack_a_data(a_data, tokenizer, pos_checking=False):
    sorts = a_data['words_sort']
    positions = a_data['position']
    input_ids = a_data['input_ids']
    result = tokenizer.convert_ids_to_tokens(sorts)
    filter = []
    for word, position in zip(result, positions):
        if checkwords(word, stopwords):
            filter.append((word, position))
    #print(filter)
    # print(len(filter))
    changesNum = int(round(len(filter) * change_porpotion, 0))
    for i in range(changesNum):
        a_pair = filter[i]
        origin_word, position = a_pair
        #origin_pos = nltk.pos_tag([origin_word])[0][1]
        #print(origin_pos)
        if not word_dict.get(origin_word, None):  # 如果词表里没有origin_word就直接跳到下一个
            continue
        if pos_checking:
            rep_list = wntools.GenerateSimWordByWNAndPosChecking(origin_word)
        else:
            rep_list = wntools.GenerateSimWordByWN(origin_word)
                #这里决定是否使用PosTagging
        if not rep_list:  # 如果WordNet没有找到同义词就跳到下一个
            continue
        else:
            results = []
            for rep_word in rep_list:
                sim_scores = cal_cos_sim(origin_word, rep_word, word_dict)
                if (not sim_scores) or (sim_scores < 0.6):
                    continue
                results.append((rep_word, sim_scores))
            if results:  # 从结果中找到符合要求的，替换原来的input_ids即可
                results.sort(reverse=True, key=lambda x: x[1])
                # print('position:', position)
                for result in results:
                    attack_word = result[0]

                    attack_word_idx = tokenizer.convert_tokens_to_ids(attack_word)  #遍历能在tokenizer词表里找到的单词
                    if attack_word_idx != 100:
                        input_ids[position] = attack_word_idx    # 更改对应位置的input_ids并推出循环
                        break
                    else:
                        continue

    a_data['input_ids'] = input_ids
    return a_data


if __name__ == '__main__':
    change_porpotion = 0.3
    start_time = time.time()
    #加载需要的东西
    word_dict = CounterFittedVectors(dict_file)
    all_data = getAttackData(attack_data_file)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    stopwords = stopwords.words('english')

    pos_checking = True  #是否使用poschecking
    #print(stopwords)
    print('loadingTime:', time.time()-start_time)

    attacked_results = []
    for a_data in tqdm(all_data):
        attacked_data = attack_a_data(a_data, tokenizer, pos_checking=pos_checking)
        attacked_results.append(attacked_data)


    with open('data/wikipassageQA/attacked_rate03_pos.json', 'w', encoding='utf-8') as writer:
        for a_data in tqdm(attacked_results):
            json.dump(a_data, writer)
            writer.write('\n')

    # #pdb.set_trace()
    # start_time = time.time()
    # word1 = 'evangelical'
    # word2 = 'forgiveness'
    # results = cal_cos_sim(word1, word2, word_dict)
    # if results:
    #     print(results)
    # else:
    #     print('词表查无此词')
    # print('CalSimTime:', time.time()-start_time)


