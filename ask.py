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
        self.ent_map = {"PERSON": "Who", "ORG": "Who", "DATE": "What", "GPE":"What"}
        self.num_questions = num_questions
        self.not_allowed_after_wh = ["it","{","[","(","\"",";",":",",", "in"] + list(self.ent_map.values())\
                            + list([x.lower() for x in self.ent_map.values()])
        self.not_allowed_headings = ["overview", "gallery", "notelist"]

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
        og_question = question
        question_before_ent = question[:ent_pos]
        question_after_ent = question[ent_pos:]
        question = question_after_ent
        if "," in question:
            if ent_pos > og_question.index(substring):
                question = question[:question.index(",")]
            else:
                return None
        if ";" in question:
            if ent_pos > og_question.index(substring):
                question = question[:question.index(";")]
            else:
                return None        
        return question_before_ent+question
        '''if "," in question:
            if substring in question:
                if question.index(",") > question.index(substring) \
                        and question.index(",") > ent_pos:
                    question = question[:question.index(",")]
            else:
                return None
        if ";" in question:
            if substring in question:
                if question.index(";") > question.index(substring) \
                        and question.index(";") > ent_pos:
                    question = question[:question.index(";")]
            else:
                return None
        return question'''
    
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
        if question.split(" ")[0] in self.ent_map.values() and question.split(" ")[1] == "the":
            question = " ".join(["Why", " ".join(question.split(" ")[1:])])
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
    num_q = int(sys.argv[2])
    input_file = sys.argv[1]
    #questions = [None]*(num_questions)
    WHquestions, YNquestions=[],[]
    q_count =0
    ask = genQuestions( "medium", num_q)
    with open(input_file, "r") as file: 
        data = file.readlines()
    data_=[]
    for line in data:
        blob= TextBlob(line)
        for sent in blob.sentences:
            data_.append(str(sent))
    data = data_
    data_=[]
    heading_candidates=[]
    article_heading = None
    end_of_headings =False
    for line in data:
        if (not end_of_headings) and len(line.split(" "))< 6 and line[-1] != ".":
            if article_heading == None:
                article_heading = line
            else:
                if "references" not in line.lower() and "notes" not in line.lower():
                    for h in ask.find_noun_phrases(parser.parse(line)):
                        clean_heading = re.sub(r'[^\s^a-z^A-Z^0-9]', ""," ".join(h.leaves()))
                        if not sum(1 for x in ask.not_allowed_headings if x in clean_heading ):
                            heading_candidates.append(clean_heading)
                            break
                else:
                    end_of_headings = True
        else:
            data_.append(line)
    data = data_
    print(heading_candidates)
    lines_of_interest = ask.find_NER_SENT(data)
    for sentence in lines_of_interest:
        try:
            q_ = ask.gen(sentence)
        except: 
            continue
        if q_ is not None:
            if(q_[0] == "WH"):
                WHquestions.append([sentence, q_[1]])
            else:
                YNquestions.append([sentence, q_[1]])
    final_list=[]
    remaining_list=[]
    num_wh_q,wh_q_count = max(1,int(0.8*num_q)), 0
    num_yn_q, yn_q_count = min(2,max(1,int(0.2*num_q))), 0
    per_heading_thresh = max(1,int(0.1*num_wh_q))
    random.shuffle(WHquestions)
    for whQ in WHquestions:
        final_list.append(whQ[1])
        wh_q_count +=1
        if num_wh_q < wh_q_count:
            break
    if(len(WHquestions)>num_wh_q):
        WHquestions = WHquestions[num_wh_q:]
    random.shuffle(YNquestions)
    for i in range(num_yn_q):
        final_list.append(YNquestions[i][1])
    #print(random.sample(["Comment on ", "Describe ", "Discuss about "],1)[0] ,\
    #            random.sample(heading_candidates[:4], 1)[0].lower(), \
    #            random.sample([" in context of ", " with regards to ", " in reference to "],1)[0],\
    #            (article_heading))
    if(num_q>10):
        hard_q = random.sample(["Comment on ", "Describe ", "Discuss about "],1)[0] +\
                random.sample(heading_candidates[:4], 1)[0].lower()+ \
                random.sample([" in context of ", " with regards to ", " in reference to "],1)[0]+\
                (article_heading)+"."
        final_list.append(hard_q)
    if len(final_list)< num_q:
        new_list = random.sample(WHquestions, max(1, int(0.7*(num_q-len(final_list)))))\
                    + random.sample(YNquestions, max(1, int(0.3*(num_q-len(final_list)))))
    
        for i in range(min(len(new_list),max(1,num_q -len(final_list)))):
            if(len(new_list[i])>1):
                final_list.append(new_list[i][1])
            else:
                final_list.append(new_list[i])
    random.shuffle(final_list)
    for q in final_list:
        print(q)
        print('\n')