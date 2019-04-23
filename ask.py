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
        self.not_allowed_after_wh = ["{","[","(","\"",";",":",",", "in"] + list(self.ent_map.values())\
                            + list([x.lower() for x in self.ent_map.values()])
        
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
    
    def find_full_dependency_phrase(self, doc, phrase):
        distances=[]
        for token in doc:
            subjString=""
            if ("sub" in token.dep_ or "obj" in token.dep_):
                tokenLeft = " ".join([token_.text for token_ in token.lefts])
                tokenRight = " ".join([token_.text for token_ in token.rights])
                subjString = " ".join([x for x in [tokenLeft, token.text, tokenRight] if not x is ""])
                distances.append([subjString, Levenshtein.distance(subjString, phrase)])
        return list(sorted(distances, key= lambda x:x[1]))[0][0] 

    def processComma(self, question, substring, ent_pos):
        if "," in question:
            if substring in question:
                if question.index(",") > question.index(substring) \
                        and question.index(",") > ent_pos:
                    question = question[:question.index(",")]
            else:
                return None
        return question
    
    def find_obj_pos(self, question, doc, tree):
        refined=""
        doc = self.nlp(question)
        ent_flag =0
        ent_phrase = ""
        for token in doc:
            if not ent_flag:
                if token.ent_type_ in self.ent_map:
                    ent_phrase += " "+token.text
                    ent_flag=1
                else:
                    refined+= " "+token.text
            else:
                if token.ent_type_ in self.ent_map:
                    ent_phrase += " "+token.text
                else:
                    break
        if ent_phrase:
            final_ent_phrase =self.find_full_phrase(tree,self.find_full_dependency_phrase(doc, ent_phrase))
            return len(refined) - 3+ len(final_ent_phrase)
        else: 
            return  -1

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

    def check_Q_grammar(self, question, substring):
        question_list = question.split(" ")
        if len(question_list)< 3:
            return False
        if sum( x in question_list[question_list.index(substring)+1] for x in self.not_allowed_after_wh):
            return False
        
        tree = parser.parse(question)
        if tree.label() == "FRAG":
            return False
        for node in tree:
            if node.label() == "FRAG":
                return False
        
        return True
    
    def genYesNo(self, sentence):
        modifier = random.choice(["true", "false"])
        sentence_ = sentence[0].lower() + sentence[1:]
        if modifier=="true":
            return "Is it "+modifier+" that "+ sentence_
        else:
            if "is" in sentence_:  
                return "Is it true that "+ sentence_[:sentence_.index(" is ")+3] + " not " + sentence_[sentence_.index(" is ")+4:]
            else:
                return "Is it "+modifier+" that "+ sentence_

    def gen(self, sentence):
        qtype = "WH"
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
        pos = self.find_obj_pos(question, doc, tree)
        question = self.processComma(question, substring, pos)
        if not question:
            return None
        if(not self.check_Q_grammar(question, substring)):
            question = self.genYesNo(sentence)
            qtype = "YN"
        if(question[-1]=="."):
            question = question[:-1] + "?"
        else:
            question +="?"
        question = re.sub(r'\s{1,}(\?)', r'\1',question)
        question = re.sub(r'[^\w|^\d](\?)', r'\1',question)
        return qtype,question


    def hasNER(self,sentence):
        doc = self.nlp(sentence)
        for word in doc:
            if word.ent_type_ in self.ent_map:
                return True
        return False
    
    def find_NER_SENT(self,data):
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
        try:
            q_ = ask.gen(sentence)
        except:
            print("error in sentence: ", sentence)
        if q_ is not None:
            #questions[q_count] = ask.gen(sentence)
            questions.append([sentence, q_])
            q_count +=1
        #if q_count == num_questions:
        #    break
    for (a,q) in questions:
        print("Context: {}\n Q:{}\n\n".format(a,q))
        print("\n")