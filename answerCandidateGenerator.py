import spacy
import wmd
import sys

#run in command line
#file1:text
#file2:questions
#returns ranking which is a list of tuples with the 5 most likely senteces to contain the answer
#first elemnt of the tuple is the sentence
#second is the similarity ranking

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
		if clean=="":
			continue
		similarity = question.similarity(nlp(clean))
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
for question in questionList:
	print(answer(question,sentences))
