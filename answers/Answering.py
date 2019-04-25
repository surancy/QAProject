#!/usr/bin/env python
# coding: utf-8

# In[210]:


import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from textblob import TextBlob


# In[211]:


# files
# what we got from the ask.py
question = "set1_a1.txt"
# what we got from the passage retrieval - first liners
candidate = "set1_a1_ansCandidate_temp.txt"

with open(question,'r') as file:
    questionList = file.readlines()
questionList = [lines.rstrip() for lines in questionList] # remove "\n"

with open(candidate,"r") as file:
    candidateList = file.readlines()
candidateList = [lines.rstrip() for lines in candidateList] # remove "\n"


# In[218]:


binaryQ = ["is", "isn't", "are", "aren't", "am", "was", "wasn't", "were", "weren't", "does", "doesn't", "do",
           "don't", "did", "didn't", "have", "havn't", "has", "hasn't", "had", "hadn't", "will", "won't", 
           "would", "wouldn't", "could", "couldn't", "can", "can't"]
binaryDct = dict(zip(binaryQ,[i for i in range(len(binaryQ))]))
ans = ""
ansList = list()


for i in range(len(questionList)):
    Q = nlp(questionList[i])
    text = nlp(candidateList[i])
    
    
    if "whom" in [w.text for w in Q] or "who" in [w.text for w in Q]:
        ans = person_ans(Q,text)
    elif binaryDct.get(str(Q[0]).lower()) != None:
        ans = binary_ans(str(text))
    elif str(Q[0]).lower() == "when":
        ans = time_ans(Q,text)
    elif str(Q[0]).lower() == "where":
        ans = loc_ans(Q,text)
    else:
        ans = general_ans(Q,text)

    ansList.append(ans)
    print("Question:       ", str(Q))
    print("Candidate text  ", str(text))
    print("Answer:         ", ans)
    print("=================")


# In[153]:


with open ("answers_s1a1.txt", "w") as output:
    for ans in ansList:
        output.write(ans+ "\n")


# In[212]:


def general_ans(Q,text):
    # Reconstruct Q
    strQbody = reconstructQ(Q)
    ans = str(text).capitalize()
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
        ans = str(longSentence).capitalize() + " " + strQbody + "."
        # if NER tags exist in the longSentence's noun phrases' root
        for chunks in longSentence.noun_chunks:
            for keys in ent_dict.keys():
                if str(chunks.root) in str(keys):
                    ans = str(keys).capitalize() + "."
                    NERflag = True
        # if no NER tags found in longSentence, reconstruct it            
        if NERflag == False:
            for chunks in longSentence.noun_chunks:
                if (str(chunks.root.head) == "of"):
                    ans = str(chunks.root.head.head).capitalize() + " " + "of " + str(chunks) + " " + strQbody + "."
                else:
                    ans = str(chunks).capitalize() + "."

    return ans


# In[213]:


# helper function to reconstruct Q
def reconstructQ(Question):
    # Q is spacy tokenized
    cleanQ = list(Q)
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


# In[214]:


def person_ans(Q,text):
    strQbody = reconstructQ(Q)
    strText = str(text).capitalize()
    rooti = 0
    ans = strText
    # default ans:
    # if no correct NER tags found in text, reconstruct it            
    # for chunks in text.noun_chunks:
    #     if (str(chunks.root.head) == "of"):
    #         ans = str(chunks.root.head.head).capitalize() + " " + "of " + str(chunks) + " " + strQbody + "....."
    #     else:
    #         ans = str(chunks) + "...."

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

    # default: find ans with the closest distance from text root
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
        ans = str(min(ent_distance, key=ent_distance.get)).capitalize() + "."

    for key in ent_person_dict.keys():
        for keys in ent_q_person:
            for chunks in text.noun_chunks:
                # if the phrase with correct NER tag is also the subject of the sentence, find the ans
                if ((str(chunks.root.dep_) == "nsubj" or str(chunks.root.dep_) == "nsubjpass") and str(chunks.root) in str(key)):
                    ans = str(chunks).capitalize() + "."
                # if none of the phrase with correct NER tag is subject, find the one with novelty
                elif (str(chunks.root) not in str(keys) and str(chunks.root) in key):
                    ans = str(chunks).capitalize() + "."
    
    return ans


# In[215]:


def binary_ans(strText):
    ans = "Yes."
    testimonial = TextBlob(strText)
    print(testimonial.sentiment.polarity)
    if testimonial.sentiment.polarity >= -0.1:
        ans = "Yes."
    else:
        ans = "No."
    return ans


# In[216]:


def time_ans(Q,text):
    # TIME/QUANTITY Ans type
    strText = str(text).capitalize()
    ans = strText
    # dict to store NER tags of the text
    ent_dict = dict()
    for X in text.ents:
        ent_dict[X.text] = X.label_
    for k,v in ent_dict.items():
        if (v == "DATE" or v == "CARDINAL" or v == "TIME" or v == "QUANTITY"):
            ans = str(k).capitalize() + "."

    return ans


# In[217]:


def loc_ans(Q,text):
    ans = str(text).capitalize()
    ent_dict_q = dict()
    for X in Q.ents:
        ent_dict_q[X.text] = X.label_
        
    ent_dict_text = dict()
    for X in text.ents:
        if (str(X.label_) == "GPE" or str(X.label_) == "LOC"):
            ent_dict_text[X.text] = X.label_
    for k,v in ent_dict_text.items():
        # if there is more than one correct NER category, apply the novelty rule
        if len(ent_dict_text) > 1:
            for key in ent_dict_text:
                if key not in ent_dict_q:
                    ans = key
                    
    return ans


# In[160]:


# =======Test case from writeup=========
text = "Pittsburgh was named in 1758 by General John Forbes, in honor of British statesman William Pitt, 1st Earl of Chatham."
# Q = "When was Pittsburgh named by General John Forbes, in honor of British statesman William Pitt, 1st Earl of Chatham?"
## ans: 1758.

# Q = "What was named in 1758 by General John Forbes?"
# ans: Pittsburgh.

# Q = "Who named Pittsburgh in 1758?"
## ans: John Forbes.

# Q = "In honor of whom was Pittsburgh named in 1758 by General John Forbes?"
## ans: British statesman william pitt.


# In[169]:


# ===== Test some gabagge input =====
text = "this is totally not what we are looking for"
Q = "this is not gonna make sense."


# In[170]:


Q = "What is the period in the third millennium also known as?"
text = "Old Kingdom of EgyptThe Old Kingdom is the period in the third millennium (c"


# In[208]:


Q = "Is it true that have CMU been built?"
text = "I don't actually hate to learn mathmetics at all, in fact, it was one of my favorite"


# In[209]:


text = nlp(text)
Q = nlp(Q)
# time_ans(Q, text) # 1758.
# general_ans(Q,text) #Pittsburgh.
# person_ans(Q,text) # 1. John Forbes. 2. British statesman william pitt.
# binary_ans(str(text))


# In[ ]:




