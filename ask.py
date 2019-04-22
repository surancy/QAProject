import benepar
import spacy
import re
import Levenshtein
nlp = spacy.load("en_core_web_sm")
import random
import nltk
import sys
from textblob import TextBlob
parser = benepar.Parser("benepar_en2")

def overlap(a, b):
    return max(i for i in range(len(b)+1) if a.endswith(b[:i]))

class genQuestions():
    def __init__(self, modelSize = "small", num_questions = 5):
        if modelSize == "small":
            self.nlp = spacy.load("en_core_web_sm")
        elif modelSize == "medium":
            self.nlp = spacy.load("en_core_web_md")
        elif modelSize == "large":
            self.nlp = spacy.load("en_core_web_lg")
        else:
            print("Invalid model size.")
        self.ent_map = {"PERSON": "Who", "ORG": "Who", "DATE": "When", "GPE":"What"}
        self.num_questions = num_questions
        
    def form_question(self, phrase, sentence):
        phrase_list = re.findall("\w", phrase)
        sentence_list = re.findall("\w", sentence)
        
    def find_noun_phrases(self,tree):
        return [subtree for subtree in tree.subtrees(lambda t: t.label()=='NP')]
        
    def find_full_phrase(self, tree, phrase):
        distances=[]
        nounPhrase = ""
        for np in self.find_noun_phrases(tree):
            nounPhrase = " ".join(np.leaves())
            distances.append([nounPhrase, Levenshtein.distance(nounPhrase, phrase)])
        return list(sorted(distances, key= lambda x:x[1]))[0][0]
    
    def refineQuestion(self, question):
        refined =""
        doc = self.nlp(question)
        ent_flag =0
        for token in doc:
            if not ent_flag:
                refined+= " "+token.text
                if token.ent_type_ in self.ent_map:
                    ent_flag=1
            else:
                if token.ent_type_ in self.ent_map:    
                    refined+= " "+token.text
                else:
                    break
        return refined

    def subjQuestion(self, doc):
        subjString = ""
        flag = 1
        for token in doc:
            if flag and ("sub" in token.dep_ or "obj" in token.dep_):
                tokenLeft = " ".join([token_.text for token_ in token.lefts])
                tokenRight = " ".join([token_.text for token_ in token.rights])
                subjString = " ".join([x for x in [tokenLeft, token.text, tokenRight] if not x is ""])
                flag = 0
            else:
                if sum(token.text == x for x in ["is", "was"]):
                    return subjString
                else:
                    phrase = " "+token.text+" " 
                    o= overlap(subjString, phrase)
                    subjString += phrase[o:]
        
    def preprocess(self, sentence):
        return (re.sub("\s*\([^]]*\)", r"" , sentence))

    def checkAKA(self, sentence):
        if "also known as" in sentence:
            return sentence[:1+sentence.lower().find("also known as")+len("also known as")]
        else:
            return sentence

    def gen(self, sentence):
        sentence = self.preprocess(sentence)
        doc = self.nlp(sentence)
        phrase = self.subjQuestion(doc)
        if not phrase: 
            return None
        tree = parser.parse(sentence)
        phrase  = self.find_full_phrase(tree, phrase)
        phrase = re.sub(r"\s+([,| \"| \'| :])", r"\1" , phrase)
        substring = "What"
        for word in doc:
            if word.text in phrase and word.ent_type_ in self.ent_map:
                substring = self.ent_map[word.ent_type_]
                break
        question = re.sub(phrase, substring+" ", sentence,1)
        if substring in question:
            shift =  question.index(substring)
            if shift != 0:
                question = question[shift:]
        question = re.sub(r'\s{2,}', ' ',question)
        question = self.checkAKA(question)
        question = self.refineQuestion(question)
        if(question[-1]=="."):
            question = question[:-1] + "?"
        else:
            question +="?"        
        return question 


    def hasNER(self,sentence):
        doc = self.nlp(sentence)
        for word in doc:
            if word.ent_type_ in self.ent_map:
                return True
        return False
    
    def find_NER_SENT(self,document):
        result =[]
        for line in data:
            doc = self.nlp(self.preprocess(line))
            for sent in doc.sents:
                if len(sent.text.split(" ")) > 5 and self.hasNER(sent.text):
                        result.append(sent.text)
        random.shuffle(result)
        return result

if __name__ == "__main__":
    num_questions = int(sys.argv[2])
    input_file = sys.argv[1]
    #questions = [None]*(num_questions)
    questions=[]
    q_count =0
    with open(input_file, "r") as file: 
        data = file.readlines()
    data_=[]
    for line in data:
       blob= TextBlob(line)
       for sent in blob.sentences:
           data_.append(str(sent))
    data = data_
    ask = genQuestions( "medium", num_questions)
    lines_of_interest = ask.find_NER_SENT(data)
    for sentence in lines_of_interest:
        q_ = ask.gen(sentence)
        if q_ is not None:
            #questions[q_count] = ask.gen(sentence)
            questions.append([sentence, ask.gen(sentence)])
            q_count +=1
        #if q_count == num_questions:
        #    break
    for (a,q) in questions:
        print("Context: {}\n Q:{}\n\n".format(a,q))
        print("\n")