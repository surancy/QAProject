#!/usr/bin/env python
# coding: utf-8

# In[17]:


import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

from textblob import TextBlob


# In[30]:


def ans(text,Q):
    """
    for who questions, look for phrases with the NER tags: PERSON, ORG, in the text that contains answer.
    only look for phrases where proper noun is the subject within sentence
    
    if no NER tag, look for similarities of word2vec
    """

    
    if (str(Q[0]).lower() == "who"):
        text = nlp(text)
        ans = ""
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for token in text:
            for k,v in ent_dict.items():
                if (token.dep_ == "nsubj"):
                    if (v == "PERSON" or v == "ORG"):
                        ans = token.text
    
    """
    for when questions, look for phrases with the NER tags: DATE, in the text that contains answer.
    
    backup: 
    """
    if (str(Q[0]).lower() == "when"):
        text = nlp(text)
        ans = ""
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for k,v in ent_dict.items():
            if (v == "DATE"):
                ans = k
    """
    for where questions, look for phrases with the NER tags: GPE, LOC, in the text that contains answer.
    """
    if (str(Q[0]).lower() == "where"):
        text = nlp(text)
        ans = ""
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for k,v in ent_dict.items():
            if (v == "GPE" or v == "LOC"):
                ans = k
                    
                    
    """
    for why questions, extract words as "because, since, as 
    //ps: as can be very erroneous. need to find additional rule" 
    reconstruct sentense with prepositional phase to make the answer grammatically correct
    """
#     if (str(Q[0]).lower() == "because" or str(Q[0]).lower() == "since" or str(Q[0]).lower() == "as"):
        

    
    """
    for yes/no questions, do the sentiment analysis
    """  
    if (str(Q[0]).lower() == "does" or str(Q[0]).lower() == "do" or str(Q[0]).lower() == "did"
        or str(Q[0]).lower() == "is" or str(Q[0]).lower() == "are" 
        or str(Q[0]).lower() == "have" or str(Q[0]).lower() == "has" or str(Q[0]).lower() == "had"):
            testimonial = TextBlob(text)
            print(testimonial.sentiment.polarity)
            if testimonial.sentiment.polarity >= 0:
                ans = "Yes"
            else:
                ans = "No"
    
    
    return ans


# In[31]:


# testing
# text = nlp("Under King Djoser, the first king of the Third Dynasty of the Old Kingdom the royal capital of Egypt was moved to Memphis, where Djoser established his court.")
# Q = nlp("who is the first king of the Third Dynasty of the Old Kingdom?")
# Q = nlp("where does the Old Kingdom the royal capital of Egypt moved to?")

text = "The Old Kingdom is the period in the third millennium (c. 2686-2181 BC) also known as the \'Age of the Pyramids\' or \'Age of the Pyramid Builders\' as it includes the great 4th Dynasty when King Sneferu perfected the art of pyramid building and the pyramids of Giza were constructed under the kings Khufu, Khafre, and Menkaure. "
# Q = nlp("who perfected the art of pyramid building?")
# Q = nlp("when is the Old Kingdom?")
Q = nlp("Is the Old Kingdom also known as the \'Age of the Pyramids\' or \'Age of the Pyramid Builders\'?")




ans(text,Q)


# In[ ]:




