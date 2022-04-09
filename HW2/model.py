from itertools import count
from logging.config import stopListening
import math
from collections import Counter, defaultdict, OrderedDict
from pydoc import doc
from typing import List
from unittest import expectedFailure
from cv2 import exp

import nltk
import numpy as np
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm


class Ngram:
    def __init__(self, config=None, n=2):
        self.tokenizer = ToktokTokenizer()
        self.n = n
        self.model = None
        self.config = config

    def tokenize(self, sentence):
        '''
        E.g.,
            sentence: 'Here dog.'
            tokenized sentence: ['Here', 'dog', '.']
        '''
        return self.tokenizer.tokenize(sentence)

    def get_ngram(self, corpus_tokenize: List[List[str]]):
        '''
        Compute the co-occurrence of each pair.
        '''
        # begin your code (Part 1)
        unigramCnt = {}
        bigramCnt  = {}
        trigramCnt = {}
        bigrams    = []
        unigrams   = []
        trigrams   = []
        total = 0
        for document in corpus_tokenize:
            for idx in range(len(document)-1):
                total += 1
                if idx != len(document) - 2:
                    trigram = (document[idx], document[idx+1], document[idx+2])
                    if trigram in trigramCnt:
                        trigramCnt[trigram] += 1
                    else:
                        trigramCnt[trigram] = 1
                bigram = (document[idx], document[idx+1])
                unigrams.append(document[idx])
                bigrams.append(bigram)
                trigrams.append(trigram)


                if bigram in bigramCnt:
                    bigramCnt[bigram] += 1
                else:
                    bigramCnt[bigram] = 1

                if document[idx] in unigramCnt:
                    unigramCnt[document[idx]] += 1
                else:
                    unigramCnt[document[idx]] = 1
        model = {}
        if self.n == 1:
            for uni in unigrams:
                x = uni
                prob = unigramCnt[x] / total
                model[x] = prob
            
            features = OrderedDict(sorted(unigramCnt.items(), key=lambda item: -item[1]))
        elif self.n == 2:
            for bigram in bigrams:
                x = bigram[0]
                y = bigram[1]
            
                prob = bigramCnt[bigram] / unigramCnt[x]
                if x in model:
                    model[x][y] = prob
                else:
                    model[x] = dict()
                    model[x][y] = prob
            
            features = OrderedDict(sorted(bigramCnt.items(), key=lambda item: -item[1]))
        elif self.n == 3:
            for trigram in trigrams:
                x = trigram[0]
                y = trigram[1]
                z = trigram[2]

                prob = trigramCnt[trigram] / bigramCnt[(x, y)]
                if (x, y) in model:
                    model[(x, y)][z] = prob
                else:
                    model[(x, y)] = dict()
                    model[(x, y)][z] = prob
            features = OrderedDict(sorted(trigramCnt.items(), key=lambda item: -item[1]))
        return model, features
        # end your code
    
    def train(self, df):
        '''
        Train n-gram model.
        '''
        corpus = [['[CLS]'] + self.tokenize(document) for document in df['review']]     # [CLS] represents start of sequence
        
        # You may need to change the outputs, but you need to keep self.model at least.
        self.model, self.features = self.get_ngram(corpus)

    def compute_perplexity(self, df_test) -> float:
        '''
        Compute the perplexity of n-gram model.
        Perplexity = 2^(entropy)
        '''
        if self.model is None:
            raise NotImplementedError("Train your model first")

        corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']]
        # begin your code (Part 2)
        # self.model, self.features = self.get_ngram(corpus)
        l = 0
        total = 0
        if self.n == 1:
            for document in corpus:
                for idx in range(len(document)):
                    x = document[idx]
                    feature = x
                    cnt = self.features[feature]
                    l += math.log2(self.model[x]) * cnt
                    total += cnt
        elif self.n == 2:
            for document in corpus:
                for idx in range(len(document) - 1):
                    x = document[idx]
                    y = document[idx+1]
                    feature = (x, y)
                    cnt = self.features[feature]
                    l += math.log2(self.model[x][y]) * cnt
                    total += cnt
        elif self.n == 3:
            for document in corpus:
                for idx in range(len(document) - 2):
                    x = document[idx]
                    y = document[idx+1]
                    z = document[idx+2]
                    feature = (x, y, z)
                    cnt = self.features[feature]
                    l += math.log2(self.model[(x, y)][z]) * cnt
                    total += cnt

        l /= total

        perplexity = pow(2, -l)

        return perplexity


    def train_sentiment(self, df_train, df_test):
        '''
        Use the most n patterns as features for training Naive Bayes.
        It is optional to follow the hint we provided, but need to name as the same.

        Parameters:
            train_corpus_embedding: array-like of shape (n_samples_train, n_features)
            test_corpus_embedding: array-like of shape (n_samples_train, n_features)
        
        E.g.,
            Assume the features are [(I saw), (saw a), (an apple)],
            the embedding of the tokenized sentence ['[CLS]', 'I', 'saw', 'a', 'saw', 'saw', 'a', 'saw', '.'] will be
            [1, 2, 0]
            since the bi-gram of the sentence contains
            [([CLS] I), (I saw), (saw a), (a saw), (saw saw), (saw a), (a saw), (saw .)]
            The number of (I saw) is 1, the number of (saw a) is 2, and the number of (an apple) is 0.
        '''
        # begin your code (Part 3)

        # step 1. select the most feature_num patterns as features, you can adjust feature_num for better score!
        feature_num = self.config['num_features']
        # step 2. convert each sentence in both training data and testing data to embedding.
        # Note that you should name "train_corpus_embedding" and "test_corpus_embedding" for feeding the model.

        self.train(df_train)
        gramIdx = {}
        gramPos = {}
        gramNeg = {}
        sumPos = 0
        sumNeg = 0
        train_corpus = [['[CLS]'] + self.tokenize(document) for document in df_train['review']]
        y = list(df_train['sentiment'])
        if self.n == 1:
            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)):
                    unigram = document[idx]
                    if y[i] == 1:
                        if unigram in gramPos:
                            gramPos[unigram] += 1
                        else:
                            gramPos[unigram] = 1
                        sumPos += 1
                    else:
                        if unigram in gramNeg:
                            gramNeg[unigram] += 1
                        else:
                            gramNeg[unigram] = 1
                        sumNeg += 1
            sumAll = sumPos + sumNeg
            allUnigram = {**gramPos, **gramNeg}
            chiFeatures = []
            for key in tqdm(allUnigram):
                keyCnt = 0
                if key in gramPos:
                    keyCnt += gramPos[key]
                if key in gramNeg:
                    keyCnt += gramNeg[key]
                e11 = sumPos * keyCnt / sumAll
                e10 = sumNeg * keyCnt / sumAll
                e01 = sumPos * (sumAll-keyCnt) / sumAll
                e00 = sumNeg * (sumAll-keyCnt) / sumAll
                chi =  ((gramPos[key] if key in gramPos else 0) - e11)**2 / e11
                chi += ((gramNeg[key] if key in gramNeg else 0) - e10)**2 / e10
                chi += (sumPos - (gramPos[key] if key in gramPos else 0) - e01)**2 / e01
                chi += (sumNeg - (gramNeg[key] if key in gramNeg else 0) - e00)**2 / e00

                chiFeatures.append((key, chi))
            chiFeatures = sorted(chiFeatures, key=lambda pair: -pair[1])
            print(chiFeatures[:20])
            kBestFeatures = chiFeatures[:feature_num]

            test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 

            train_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_train['review']))]
            test_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_test['review']))]

            for i, pair in enumerate(kBestFeatures):
                gramIdx[pair[0]] = i
            

            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)):
                    unigram = document[idx]
                    if unigram in gramIdx:
                        train_corpus_embedding[i][gramIdx[unigram]] += 1

            for i, document in enumerate(test_corpus):
                for idx in range(len(document)):
                    unigram = document[idx]
                    if unigram in gramIdx:
                        test_corpus_embedding[i][gramIdx[unigram]] += 1
        elif self.n == 2:
            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)-1):
                    bigram = (document[idx], document[idx+1])
                    if y[i] == 1:
                        if bigram in gramPos:
                            gramPos[bigram] += 1
                        else:
                            gramPos[bigram] = 1
                        sumPos += 1
                    else:
                        if bigram in gramNeg:
                            gramNeg[bigram] += 1
                        else:
                            gramNeg[bigram] = 1
                        sumNeg += 1
            sumAll = sumPos + sumNeg
            allBigram = {**gramPos, **gramNeg}
            chiFeatures = []
            for key in tqdm(allBigram):
                keyCnt = 0
                if key in gramPos:
                    keyCnt += gramPos[key]
                if key in gramNeg:
                    keyCnt += gramNeg[key]
                e11 = sumPos * keyCnt / sumAll
                e10 = sumNeg * keyCnt / sumAll
                e01 = sumPos * (sumAll-keyCnt) / sumAll
                e00 = sumNeg * (sumAll-keyCnt) / sumAll
                chi =  ((gramPos[key] if key in gramPos else 0) - e11)**2 / e11
                chi += ((gramNeg[key] if key in gramNeg else 0) - e10)**2 / e10
                chi += (sumPos - (gramPos[key] if key in gramPos else 0) - e01)**2 / e01
                chi += (sumNeg - (gramNeg[key] if key in gramNeg else 0) - e00)**2 / e00

                chiFeatures.append((key, chi))
            chiFeatures = sorted(chiFeatures, key=lambda pair: -pair[1])
            print(chiFeatures[:20])
            kBestFeatures = chiFeatures[:feature_num]

            test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 

            train_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_train['review']))]
            test_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_test['review']))]

            for i, pair in enumerate(kBestFeatures):
                gramIdx[pair[0]] = i
            

            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)-1):
                    bigram = (document[idx], document[idx+1])
                    if bigram in gramIdx:
                        train_corpus_embedding[i][gramIdx[bigram]] += 1

            for i, document in enumerate(test_corpus):
                for idx in range(len(document)-1):
                    bigram = (document[idx], document[idx+1])
                    if bigram in gramIdx:
                        test_corpus_embedding[i][gramIdx[bigram]] += 1
        elif self.n == 3:
            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)-2):
                    trigram = (document[idx], document[idx+1], document[idx+2])
                    if y[i] == 1:
                        if trigram in gramPos:
                            gramPos[trigram] += 1
                        else:
                            gramPos[trigram] = 1
                        sumPos += 1
                    else:
                        if trigram in gramNeg:
                            gramNeg[trigram] += 1
                        else:
                            gramNeg[trigram] = 1
                        sumNeg += 1
            sumAll = sumPos + sumNeg
            allTrigram = {**gramPos, **gramNeg}
            chiFeatures = []
            for key in tqdm(allTrigram):
                keyCnt = 0
                if key in gramPos:
                    keyCnt += gramPos[key]
                if key in gramNeg:
                    keyCnt += gramNeg[key]
                e11 = sumPos * keyCnt / sumAll
                e10 = sumNeg * keyCnt / sumAll
                e01 = sumPos * (sumAll-keyCnt) / sumAll
                e00 = sumNeg * (sumAll-keyCnt) / sumAll
                chi =  ((gramPos[key] if key in gramPos else 0) - e11)**2 / e11
                chi += ((gramNeg[key] if key in gramNeg else 0) - e10)**2 / e10
                chi += (sumPos - (gramPos[key] if key in gramPos else 0) - e01)**2 / e01
                chi += (sumNeg - (gramNeg[key] if key in gramNeg else 0) - e00)**2 / e00

                chiFeatures.append((key, chi))
            chiFeatures = sorted(chiFeatures, key=lambda pair: -pair[1])
            print(chiFeatures[:20])
            kBestFeatures = chiFeatures[:feature_num]

            test_corpus = [['[CLS]'] + self.tokenize(document) for document in df_test['review']] 

            train_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_train['review']))]
            test_corpus_embedding = [[0] * len(kBestFeatures) for _ in range(len(df_test['review']))]

            for i, pair in enumerate(kBestFeatures):
                gramIdx[pair[0]] = i
            

            for i, document in tqdm(enumerate(train_corpus)):
                for idx in range(len(document)-2):
                    trigram = (document[idx], document[idx+1], document[idx+2])
                    if trigram in gramIdx:
                        train_corpus_embedding[i][gramIdx[trigram]] += 1

            for i, document in enumerate(test_corpus):
                for idx in range(len(document)-2):
                    trigram = (document[idx], document[idx+1], document[idx+2])
                    if trigram in gramIdx:
                        test_corpus_embedding[i][gramIdx[trigram]] += 1
        # end your code
        # feed converted embeddings to Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(train_corpus_embedding, df_train['sentiment'])
        y_predicted = nb_model.predict(test_corpus_embedding)
        precision, recall, f1, support = precision_recall_fscore_support(df_test['sentiment'], y_predicted, average='macro', zero_division=1)
        precision = round(precision, 4)
        recall = round(recall, 4)
        f1 = round(f1, 4)
        print(f"F1 score: {f1}, Precision: {precision}, Recall: {recall}")


if __name__ == '__main__':
    '''
    Here is TA's answer of part 1 for reference only.
    {'a': 0.5, 'saw: 0.25, '.': 0.25}

    Explanation:
    (saw -> a): 2
    (saw -> saw): 1
    (saw -> .): 1
    So the probability of the following word of 'saw' should be 1 normalized by 2+1+1.

    P(I | [CLS]) = 1
    P(saw | I) = 1; count(saw | I) / count(I)
    P(a | saw) = 0.5
    P(saw | a) = 1.0
    P(saw | saw) = 0.25
    P(. | saw) = 0.25
    '''

    # unit test
    test_sentence = {'review': ['I saw a saw saw a saw.']}
    model = Ngram(2)
    model.train(test_sentence)
    print(model.model['saw'])
    print("Perplexity: {}".format(model.compute_perplexity(test_sentence)))
