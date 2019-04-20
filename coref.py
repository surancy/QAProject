import sys
import spacy

#run in terminal
#requires spacy 2.0.12
#python3 coref.py file1.txt file2.txt
#file1=newline separated words whose coreferences you want to find
#file2=text that you want to decode

#it returns a dictionary called refs that has the words inputted in file as 
#keys and the values are a list of strings which are its coreferences

#this spacy package treats words as its own class named clusters, this
#this cleanup function turns them back to regular python strings
def cleanUpQuery(clusterList):
	ans=[]
	for c in clusterList:
		ans.append(c.text)
	return ans

#Starts out the dictionary and sets all corefs to zero for every word provided
refs=dict()
refsCluster=dict()
query = open(sys.argv[1],'r')
for line in query:
	line=line.strip()
	refs[line]=0
	refsCluster[line]=0

#this is the biggest bottleneck in the program, this loads the neural network
#that does all of the coreferencing
nlp = spacy.load('en_coref_md')
text=""
inputText=open(sys.argv[2],'r')
for line in inputText:
	text+=line
doc = nlp(text)

#clusters contains a list of of all the words and their coreferences
clusters=doc._.coref_clusters
for cluster in clusters:
	if cluster.main.text in refs:
		refsCluster[cluster.main]=cluster.mentions
		corefList=cleanUpQuery(cluster.mentions)
		refs[cluster.main.text]=corefList
#refs is the final dictionary with the keys being the list of words inputted 
#and the values being the coreferences

refFound=set()
for ref in refs:
	if refs[ref]!=0:
		refFound.add(ref)

mentions=dict()
sentenceList=doc._.coref_resolved.split(".")
for sentence in sentenceList:
	for ref in refFound:
		if ref in sentence:
			if ref not in mentions:
				mentions[ref]=[sentence]
			else:
				if sentence not in mentions[ref]:
					mentions[ref].append(sentence.strip())
print(mentions)
print(refs)