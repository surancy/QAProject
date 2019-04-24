import spacy
import wmd
import sys


nlp = spacy.load('en_core_web_md')
nlp.add_pipe(wmd.WMD.SpacySimilarityHook(nlp), last=True)
question = nlp("Is it true that egyptians in this era worshipped their Pharaoh as a god, believing that he ensured the annual flooding of the Nile that was necessary for their crops?")

def removeLast(tl,newItem):
	tl.append(newItem)
	tl=sorted(tl,key=lambda x:x[1])
	return tl[:len(tl)-1]

query = open(sys.argv[1],'r')
text=""
for line in query:
	line=line.strip()
	text+=line

ranking=[]
splitText=text.split(".")
for i in range(len(splitText)):
	clean=splitText[i].strip()
	if clean=="":
		continue
	similarity = question.similarity(nlp(clean))
	if i>4:
		ranking=removeLast(ranking,(clean,similarity))
	else:
		ranking.append((clean,similarity))

print(ranking)
#print(doc1.similarity(doc2))
