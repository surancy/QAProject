import spacy
import wmd
import sys
#run in command line
#file1:text
#file2:questions
#returns ranking which is a list of tuples with the 5 most likely senteces to contain the answer
#first elemnt of the tuple is the sentence
#second is the similarity ranking
import sys, os
from textblob import TextBlob

#warnings.simplefilter('ignore')
def removeLast(tl,newItem):
    tl.append(newItem)
    tl=sorted(tl,key=lambda x:x[1])
    return tl[:len(tl)-1]

def answer(question,text):
    ranking=[]
    question = nlp(question)
    mostSimilar=("",None)
    for i in range(len(splitText)):
        clean=splitText[i].strip()
        try:
            similarity = question.similarity(nlp(clean))
        except:
            similarity=100000
        if i>4:
            ranking=removeLast(ranking,(clean,similarity))
        else:
            ranking.append((clean,similarity))
    return ranking

nlp = spacy.load('en_core_web_md')
nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
inputText = open(sys.argv[1],'r')
text=""
for line in inputText:
    line=line.strip()
    text+=line

sentences = list(nlp(text).sents)
splitText = text.split(".")
questionList = open(sys.argv[2],'r')
answerList=[]
questions=[]
for question in questionList:
    questions.append(question)
    answerList.append(answer(question,sentences)[0][0])



def questionClassification(questionList, candidateList):
    binaryQ = ["is", "isn't", "are", "aren't", "am", "was", "wasn't", "were", "weren't", "does", "doesn't", "do",
           "don't", "did", "didn't", "have", "havn't", "has", "hasn't", "had", "hadn't", "will", "won't", 
           "would", "wouldn't", "could", "couldn't", "can", "can't"]
    binaryDct = dict(zip(binaryQ,[i for i in range(len(binaryQ))]))
    ans = ""


    for i in range(len(questionList)):
        Q = nlp(questionList[i])
        text = nlp(candidateList[i])


        if "whom" in [w.text.lower() for w in Q] or "who" in [w.text.lower() for w in Q]:
            ans = person_ans(Q,text)
        elif binaryDct.get(str(Q[0]).lower()) != None:
            ans = binary_ans(str(text))
        elif str(Q[0]).lower() == "when":
            ans = time_ans(Q,text)
        elif str(Q[0]).lower() == "where":
            ans = loc_ans(Q,text)
        else:
            ans = general_ans(Q,text)
        # print the answer
        print(ans)
        print('\n')


def general_ans(Q,text):
    # Reconstruct Q
    strQbody = reconstructQ(Q)
    # default ans
    ans = str(text)[:1].upper() + str(text)[1:]
    stoppingIndex = 0
    strText = str(text)
    ent_dict = dict()
    n = 1
    i = 0
    for X in text.ents:
        ent_dict[X.text] = X.label_
    patternFlag = False
    # find the Q pattern in text
    for i,word in enumerate(strText.split()):
        # if pattern found
        if (strText.split()[i] == strQbody.split()[0] and strText.split()[i+1] == strQbody.split()[1]):
            stoppingIndex = i
            patternFlag = True
    longSentence = " ".join(strText.split()[0:stoppingIndex])
    longSentence = nlp(longSentence)
    if (patternFlag):
        NERflag = False
        # default ans when the pattern matches
        ans = str(longSentence)[:1].upper() + str(longSentence)[1:] + " " + strQbody + "."
        # if NER tags exist in the longSentence's noun phrases' root
        for chunks in longSentence.noun_chunks:
            for keys in ent_dict.keys():
                if str(chunks.root) in str(keys):
                    ans = str(keys)[:1].upper() + str(keys)[1:] + "."
                    NERflag = True
        # if no NER tags found in longSentence, reconstruct it            
        if NERflag == False:
            for chunks in longSentence.noun_chunks:
                if (str(chunks.root.head) == "of"):
                    ans = str(chunks.root.head.head)[:1].upper() + str(chunks.root.head.head)[1:]  + " " + "of " + str(chunks) + " " + strQbody + "."
                else:
                    ans = str(chunks)[:1].upper() + str(chunks)[1:] + "."

    return ans

# helper function to reconstruct Q
def reconstructQ(Question):
    # Q is spacy tokenized
    cleanQ = list(Question)
    # Reconstruct Q
    ## get rid of question mark at the end, if any
    cleanQ = [str(words) for words in cleanQ if str(words) != "?"]
    ## fix the negation tokenization
    for i,token in enumerate(cleanQ):
        if token == "'s":
            cleanQ.remove(token)
            cleanQ[i - 1] = cleanQ[i - 1] + token
    # get rid of the "what" question head
    strQbody = " ".join(cleanQ[i] for i in range(1, len(cleanQ)))
    
    return strQbody

def person_ans(Q,text):
    strQbody = reconstructQ(Q)
    strText = str(text).capitalize()
    rooti = 0
    # default ans
    ans = str(text)[:1].upper() + str(text)[1:]

    ent_dict = dict()
    ent_person_dict = dict()
    ent_dict_q = dict()
    ent_q_person = list()

    for X in text.ents:
        ent_dict[X.text] = X.label_
    for k,v in ent_dict.items():
        if (v == "PERSON" or v == "ORG" or v == "NORP"):
            ent_person_dict[k] = v

    for X in Q.ents:
        ent_dict_q[X.text] = X.label_

    for key in ent_person_dict:
        for k in ent_dict_q:
            if key == k:
                ent_q_person.append(key)

    # default NER: find ans with the closest distance from text root
    Qroot = "".join(str(token) for token in Q if token.dep_ == "ROOT")

    lst = strText.split()
    distance = {}
    for i,token in enumerate(lst):
        if (token == Qroot):
            rooti = i

    for i,token in enumerate(lst):
        distance[token] = i
        distance[token] = distance[token] - rooti

    ent_distance = {}
    for token,dis in distance.items():
        for ent,tag in ent_person_dict.items():
            if token in ent:
                ent_distance[ent] = distance[token]
    if (len(ent_distance) != 0):
        ans = str(min(ent_distance, key=ent_distance.get))[:1].upper() + str(min(ent_distance, key=ent_distance.get))[1:] + "."

    # go for the NER with person tags
    for key in ent_person_dict.keys():
        ans = str(key)[:1].upper() + str(key)[1:] + "."
        
#     print(ent_person_dict)
    for key in ent_person_dict.keys():
        for keys in ent_q_person:
            for chunks in text.noun_chunks:
                # if the phrase with correct NER tag is also the subject of the sentence, get the ans
                if ((str(chunks.root.dep_) == "nsubj" or str(chunks.root.dep_) == "nsubjpass") and str(chunks.root) in str(key)):
                    ans = str(chunks)[:1].upper() + str(chunks)[1:] + "."
                # if none of the phrase with correct NER tag is subject, find the one with novelty
                elif (str(chunks.root) not in str(keys) and str(chunks.root) in key):
                    ans = str(chunks)[:1].upper() + str(chunks)[1:] + "."
    
    return ans

def binary_ans(strText):
    ans = "Yes."
    testimonial = TextBlob(strText)
#     print(testimonial.sentiment.polarity)
    if testimonial.sentiment.polarity >= -0.1:
        ans = "Yes."
    else:
        ans = "No."
    return ans

def time_ans(Q,text):
    # TIME/QUANTITY Ans type
    strText = str(text)
    ans = str(text)[:1].upper() + str(text)[1:]
    # dict to store NER tags of the text
    ent_dict = dict()
    for X in text.ents:
        ent_dict[X.text] = X.label_
    for k,v in ent_dict.items():
        if (v == "DATE" or v == "CARDINAL" or v == "TIME" or v == "QUANTITY"):
            ans = str(k)[:1].upper() + str(k)[1:] + "."
    
    return ans
    

def loc_ans(Q,text):
    # default ans
    ans = str(text)[:1].upper() + str(text)[1:]
    strText = str(text)
    ent_dict = dict()
    ent_person_dict = dict()
    ent_dict_q = dict()
    ent_q_person = list()

    for X in text.ents:
        ent_dict[X.text] = X.label_
    for k,v in ent_dict.items():
        if (v == "LOC" or v == "GPE"):
            ent_person_dict[k] = v

    for X in Q.ents:
        ent_dict_q[X.text] = X.label_

    for key in ent_person_dict:
        for k in ent_dict_q:
            if key == k:
                ent_q_person.append(key)

    # go for the NER with LOCATION tags
    for key in ent_person_dict.keys():
        ans = str(key)[:1].upper() + str(key)[1:] + "."
    # if more than one NER, apply novelty rule
    for key in ent_person_dict.keys():
        for keys in ent_q_person:
            for chunks in text.noun_chunks:
                if (str(chunks.root) not in str(keys) and str(chunks.root) in key):
                    ans = str(chunks)[:1].upper() + str(chunks)[1:] + "."
    
    return ans

# ====== generates and prints out the answer ======
questionClassification(questions,answerList)


