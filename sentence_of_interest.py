import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import os
import sys
from collections import Counter
import math
import en_core_web_sm
# import pdb
# from nltk.parse import stanford
import benepar


# os.environ['STANFORD_PARSER'] = 'C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/stanford-parser.jar'
# os.environ['STANFORD_MODELS'] = 'C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/stanford-parser-3.9.2-models.jar'
# java_path = "D:/Java/bin/java.exe"
# os.environ['JAVA_HOME'] = java_path
# nltk.internals.config_java(java_path)
# nltk.download("punkt")
def read_docs(data):
    """
    return : a list of set, every set contains all the words in this doc
    """
    result = []
    for s in data:
        for f in os.listdir(s):
            if f[-3:] == "txt":
                with open(s+"/"+f,encoding='utf-8',mode = "r") as _f:
                    content = _f.read()
                    result.append( set(word_tokenize(content)))
    return result

def computeTFIDF(none_len, freq_dict, data):
    docs = read_docs(data)
    N = len(docs)
    scores = []
    l = none_len
    c = freq_dict.copy()
    for k,v in c.items():
        tf_score = v/l
        count = sum([k in temp for temp in docs])
        idf_score = math.log(N/(count+1))
        tfidf_score = tf_score*idf_score
        c[k] = tfidf_score
    return c

def find_sentences_of_interest(train):
    """
    Return: a list of sentence of interest, where each element is of structure:
    (sentence Tree(is a nltk.tree.tree), dict{NER:tag}, heuristic socre)
    for example: first element of the returned list acquired from set1/a1.txt is
        (Tree('S', [Tree('NP', [Tree('DT', ['the']), Tree('NNP', ['old']), 
        Tree('NNP', ['kingdom'])]), Tree('VP', [Tree('VBZ', ['is']), 
        Tree('NP', [Tree('NP', [Tree('DT', ['the']), Tree('NN', ['period'])]), 
        Tree('PP', [Tree('PP', [Tree('IN', ['in']), Tree('NP', [Tree('DT', ['the']),
         Tree('JJ', ['third']), Tree('NN', ['millennium'])])]), 
         Tree('PRN', [Tree('-LRB-', ['-LRB-']), Tree('NP', [Tree('.', ['c.']),
          Tree('CD', ['2686-2181']), Tree('NNP', ['bc'])]), Tree('-RRB-', ['-RRB-'])])]), 
          Tree('VP', [Tree('ADVP', [Tree('RB', ['also'])]), Tree('VBN', ['known']), 
          Tree('PP', [Tree('IN', ['as']), Tree('NP', [Tree('NP', [Tree('NP', [Tree('DT', ['the']), 
          Tree('NN', ["'age"])]), Tree('PP', [Tree('PP', [Tree('IN', ['of']), 
          Tree('NP', [Tree('DT', ['the']), Tree('NNS', ['pyramids'])])]),
           Tree("''", ["'"])])]), Tree('CC', ['or']), Tree('NP', [Tree('NN', ["'age"])]), 
           Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('DT', ['the']), 
           Tree('NN', ['pyramid']), Tree('NNS', ['builders'])])]), Tree("''", ["'"])])]),
            Tree('SBAR', [Tree('IN', ['as']), Tree('S', [Tree('NP', [Tree('PRP', ['it'])]), 
            Tree('VP', [Tree('VBZ', ['includes']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), 
            Tree('JJ', ['great']), Tree('JJ', ['4th']), Tree('NNP', ['dynasty'])]), 
            Tree('SBAR', [Tree('WHADVP', [Tree('WRB', ['when'])]), 
            Tree('S', [Tree('S', [Tree('NP', [Tree('NNP', ['king']), Tree('NNP', ['sneferu'])]), 
            Tree('VP', [Tree('VBD', ['perfected']), Tree('NP', [Tree('NP', [Tree('DT', ['the']), 
            Tree('NN', ['art'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NN', ['pyramid']), 
            Tree('NN', ['building'])])])])])]), Tree('CC', ['and']), Tree('S', [Tree('NP', [Tree('NP', [Tree('DT', ['the']),
             Tree('NNS', ['pyramids'])]), Tree('PP', [Tree('IN', ['of']), Tree('NP', [Tree('NNP', ['giza'])])])]), 
             Tree('VP', [Tree('VBD', ['were']), Tree('VP', [Tree('VBN', ['constructed']), Tree('PP', [Tree('IN', ['under']), 
             Tree('NP', [Tree('DT', ['the']), Tree('NNS', ['kings']), Tree('NNP', ['khufu']), Tree(',', [',']), Tree('NNP', ['khafre']),
              Tree(',', [',']), Tree('CC', ['and']), Tree('NNP', ['menkaure'])])])])])])])])])])])])])])]), Tree('.', ['.'])]), 
              {'4th': 'ORDINAL', '2686-2181': 'DATE'}, 2)
              

        str = " ".join(sent.leaves()) 
        should give you the whole sentence, where "sent" is the tree structure shown above
    """
# train = sys.argv[1]
# n_questions = int(sys.argv[2])
#----------------------------------------read docs----------------------------------------------------------------
# d is dict where key is the title and value is a list of sentence
    d = dict()
    s = ""
    with open(train,encoding='utf-8',mode = 'r') as _f:
        for i, line in enumerate(_f):
            if i == 0:
                title = line.strip().lower()
            # delete content after "see also"
            elif line.strip().lower() in set(["see also",'references']):
                break
            else:
                s += line.strip().lower()
    d[title] = sent_tokenize(s)


    #----------------------------------------parse tree----------------------------------------------------------------
    # parse tree : select NP-VP structured sentence
    candidate = []
    # parser = stanford.StanfordParser(model_path="C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/englishPCFG.ser.gz",encoding='utf8')
    parser = benepar.Parser("benepar_en2")
    # sentences = parser.raw_parse_sents(('Hello,My name is completely Melro.','Are you ok?'))
    # pdb.set_trace()
    for v in d.values():
        # sentences = parser.raw_parse_sents(v)
        sentences = parser.parse_sents(v)
        for sentence in sentences:
            if sentence.label()=="S": # start 
                for i in range(len(sentence) - 1):
                    if sentence[i].label() == "NP" and sentence[i+1].label() == "VP":
                        candidate.append(sentence) #save this NP-VP sentence as a tree structure
                        break


    #----------------------------------------TF-IDF----------------------------------------------------------------
    Nones = set(["NN","NNS","NNP","NNPS"])
    #extract None
    #  a list of freq dict for each doc
    freq_dict = []
    # number of none tokens for each doc
    none_len = []   

    t = []
    for sent in candidate: #sent is a tree  
        for word, tag in sent.pos(): # POS
            if tag in Nones:
                t.append(word)
    none_len = len(t)
    freq_dict = Counter(t)


    dev_data = ['set1','set2','set3','set4','set5']
    # return a tf_idf dict, word:score
    # pdb.set_trace()
    tf_idf = computeTFIDF(none_len, freq_dict, dev_data)
    scores = [] #socre for every sentence
    for sent in candidate: #sent is a tree  
        for word, tag in sent.pos(): # POS
            score = 0
            if tag in Nones:
                score += tf_idf[word]
        scores.append(score)
        

    #----------------------------------------NER tag----------------------------------------------------------------
    # this is all the NER tags, delete unnecessary ones to looking for only what we need
    # NER = set(["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT",\
    #     "WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"])
    #compute NER for all candidate sentences
    NER = {"PERSON", "ORG", "DATE", "GPE"}

    # pdb.set_trace()
    # heuristic weight for tf-idf and NER
    alpha, beta = 1, 1
    candidate2 = []
    nlp = en_core_web_sm.load()
    for i, sent in enumerate(candidate):
        temp = []
        str = " ".join(sent.leaves())
        x = nlp(str)
        # pprint([(X.text, X.label_) for X in x.ents])
        for X in x.ents:
            label = X.label_
            if label in NER:# contains NER tag that we want
                #each sentence is store as a triplet (sent tree, NER tag dict, score)
                temp.append((sent, dict([(X.text, X.label_) for X in x.ents]), alpha*scores[i] + beta*len(x.ents)))
                break
        candidate2.append(temp)
    
    return candidate2

