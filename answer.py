"""
...changing the question classification criteria
usage: text is the text of which can be extracted answers from. Q is the question.
Some test cases are included at the bottom.
====In progress===
Updates: 04/20 9:00 pm
Updates on Who and When questions to formalize the answer (i.e. a full sentence instead of single words)
Refer to the testing section to see demo
"""

import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

from textblob import TextBlob # used for sentiment analysis

def ans(text,Q):
    # str copy
    strQ = str(Q)
    strText = str(text)
    # parse the Q and text with spacy parsing
    Q = nlp(Q)
    text = nlp(text)
    # initialize ans
    ans = ""
    """
    WHO Question:
    Look for phrases with the NER tags: [PERSON || ORG] in the text that contains answer
    only look for phrases where proper noun is the subject within sentence.
    Process the Question to formalize the Answer.
    """
    if (str(Q[0]).lower() == "who"):
        # Question processing
        cleanQ = list(Q)
        ## get rid of question mark at the end, if any
        cleanQ = [str(words) for words in cleanQ if str(words) != "?"]
        ## start from 1 to get rid of "who"
        ## fix the negation tokenization
        for i,token in enumerate(cleanQ):
            if token == "'s":
                cleanQ.remove(token)
                cleanQ[i - 1] = cleanQ[i - 1] + token
        strQbody = " ".join(cleanQ[i] for i in range(1, len(cleanQ)))
        
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for token in text:
            for k,v in ent_dict.items():
                if (token.dep_ == "nsubj"):
                    if (v == "PERSON" or v == "ORG"):
                        ans = str(token.text).capitalize() + " " + strQbody + "."
    
    """
    WHEN Question:
    Look for phrases with the NER tags: [DATE || TIME || CARDINAL] from the text that contains answer.
    Use dependency parsing to formalize the answer.
    """
    if (str(Q[0]).lower() == "when"):
        ans = ""
        # dict to store NER tags of the text
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for k,v in ent_dict.items():
            if (v == "DATE" or v == "CARDINAL" or v == "TIME"):
                ans = k

        token_text = textStr.split()
        for i,words in enumerate(token_text):
            if token_text[i] == ans and "BC" in token_text[i+1]:
                ans = ans + " BC"
            if token_text[i] == ans and "AD" in token_text[i+1]:
                ans = ans + " AD"

        # Dependency parsing for noun phrases
        for chunk in Q.noun_chunks:
            if chunk.root.dep_ == "nsubj":
                ans = str(chunk.text).capitalize() + " " + str(chunk.root.head.text) + " in the period of " + ans + "."
    """
    WHERE Question:
    Look for phrases with the NER tags: GPE, LOC, in the text that contains answer.
    """
    if (str(Q[0]).lower() == "where"):
        ent_dict = dict()
        for X in text.ents:
            ent_dict[X.text] = X.label_
        for k,v in ent_dict.items():
            if (v == "GPE" or v == "LOC"):
                ans = k
                    
                    
    """
    WHY Questions:
    Extract words as "because, since, as 
    //ps: as can be very erroneous. need to find additional rule" 
    reconstruct sentense with prepositional phase to make the answer grammatically correct
    """
#     if (str(Q[0]).lower() == "because" or str(Q[0]).lower() == "since" or str(Q[0]).lower() == "as"):
        

    
    """
    yes/no Questions:
    Do the sentiment analysis using Textblob
    """  
    if (str(Q[0]).lower() == "does" or str(Q[0]).lower() == "do" or str(Q[0]).lower() == "did"
        or str(Q[0]).lower() == "is" or str(Q[0]).lower() == "are" 
        or str(Q[0]).lower() == "have" or str(Q[0]).lower() == "has" or str(Q[0]).lower() == "had"):
            testimonial = TextBlob(strText)
            print(testimonial.sentiment.polarity)
            if testimonial.sentiment.polarity >= 0:
                ans = "Yes"
            else:
                ans = "No"
    
    return ans


# ===========Testing============= #
## WHO
text = "Queen Hatshepsut concentrated on expanding Egypt's external trade by sending a commercial expedition to the land of Punt."
Q = "who concentrated on expanding Egypt's external trade?"
### ans: "Hatshepsut concentrated on expanding Egypt's external trade."

## WHEN
### simple Q
text = ('The Old Kingdom is the period in'
        'the third millennium (c. 2686-2181 BC)'
        'also known as the \'Age of the Pyramids\' or \'Age of the Pyramid Builders\''
        'as it includes the great 4th Dynasty when King Sneferu perfected the art of pyramid building'
        'and the pyramids of Giza were constructed under the kings Khufu, Khafre, and Menkaure.')
Q = "when is the Old Kingdom?"
### ans: 'The old kingdom is in the period of 2686-2181 BC.'

text = "During the reign of Thutmose III (c. 1479–1425 BC), Pharaoh, originally referring to the king's palace, became a form of address for the person who was king"
Q = "when is the reign of Thutmose?"
### ans: 'The reign is in the period of 1479–1425.'

### slightly more complicated Q
text = ("The severity of the difficulties is indicated by the fact that the first known labor strike" 
        "in recorded history occurred during the 29th year of Ramesses III's reign, when the food rations for Egypt's favored"
        "and elite royal tomb-builders and artisans in the village of Deir el Medina could not be provisioned.")
Q = "when is the Ramesses's reign?"
### ans: 'The first known labor strike is in the period of the 29th year.'

ans(text,Q)

