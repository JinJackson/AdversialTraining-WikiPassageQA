from nltk.corpus import wordnet as wn
import nltk


class WNtools():
    def changePos(self, pos):
        if pos == 'NNS':
            return 'NN'
        elif pos in ['JJR', 'JJS']:
            return 'JJ'
        elif pos == 'NNPS':
            return 'NNP'
        elif pos == 'RBR' or pos == 'RBS':
            return 'RB'
        elif pos in ['VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return 'VB'
        else:
            return pos


    def GenerateSimWordByWN(self, word):
        word_synset = []
        word_synset.append(word)
        synsets = wn.synsets(word)
        for synset in synsets:
            words = synset.lemma_names()
            for w in words:
                if not (w == word_synset[0]):
                    word_synset.append(w)
        return word_synset[1:]


    def GenerateSimWordByWNAndPosChecking(self, word):
        origin_pos = nltk.pos_tag([word])[0][1]
        origin_pos = self.changePos(origin_pos)
        #print(origin_pos)
        word_synset = []
        word_synset.append(word)
        synsets = wn.synsets(word)
        for synset in synsets:
            words = synset.lemma_names()
            for w in words:
                if not (w == word_synset[0]):
                    pos = nltk.pos_tag([w])[0][1]
                    pos = self.changePos(pos)
                    #print(pos)
                    if pos == origin_pos:
                        word_synset.append(w)
        return word_synset[1:]

# wntools = WNtools()
# word = 'kill'
# print(wntools.GenerateSimWordByWNAndPosChecking(word))