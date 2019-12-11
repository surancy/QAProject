# Question Answering System

CMU 15611 - Natural Language Processing. [course website](http://demo.clab.cs.cmu.edu/NLP/).   
View the video link for final report and documentation: [report video](https://www.youtube.com/watch?v=S4rdPr5gBoQ&t=146s)

This is the IR-based QA System. Performs NLP Task of Information Retrieval from given Wiki articles. This program automatically generates questions from given articles, and answers question based on the query and the article. Built the full question generation and question answering pipeline from document parsing, passage retrieval to question generation and question answering. Parsed documents across various topics and was able to generate questions and answers that are fluent, reasonable, grammatical and idiomatic English based on the information in the document. Generated questions and answers by calculating document similarities from using word mover's distance. Performed syntax parsing, including parts-of-speech (POS) tagging and named entity recognition (NER) classification to extract meaningful keywords from documents. Constructed semantic analysis, including dependency parsing to reconstruct sentence grammar to make fluent and reasonable questions and answers.

## Pipeline

### Question generation

Question generation sampler and samples:  
Inputs:   
•	WH questions  
•	YN questions  
•	Descriptive questions  
•	Required number of questions . 
Method:   
Sample WH and YN questions in the ratio 2:8. Include 1 descriptive questions for every 10 other questions  
Output:  
A shuffled list of questions with length equal to required number of questions. 

#### Questioning pipeline

![q1]https://github.com/surancy/QAProject/blob/master/pipeline/q1.png

##### breaking down

1. **Preprocessing**  

![preprocess](https://github.com/surancy/QAProject/blob/master/pipeline/q-preprocess.png)

2. **Candidate Generation**

![candidate](https://github.com/surancy/QAProject/blob/master/pipeline/q-candidate.png)

3. **Headline Extraction**

![headline](https://github.com/surancy/QAProject/blob/master/pipeline/q-headline.png)

4. **WH-Question Module**

![wh-q](https://github.com/surancy/QAProject/blob/master/pipeline/q-wh.png)

5. **Refining**

![refine](https://github.com/surancy/QAProject/blob/master/pipeline/q-refine.png)

6. **Y/N-Question Module**

![yn](https://github.com/surancy/QAProject/blob/master/pipeline/q-yn.png)

7. **Descriptive Module**

![des](https://github.com/surancy/QAProject/blob/master/pipeline/q-descriptive.png)



### Answer Generation

Passage Retrieval > Word Mover’s Distance > Question Classification (Wh-, Binary, Other) .  
Answer Types([PERSON],[LOCATION],[TIME],[OBJECT],[BINARY],[DESCRIPTION]/[REASON])    

#### Question Answering Pipeline

![ans](https://github.com/surancy/QAProject/blob/master/pipeline/ans.png)

##### breaking down

![ans-one](https://github.com/surancy/QAProject/blob/master/pipeline/ans1.png)

1. **Passage Retrieval**  

- Word Mover’s Distance[1]  
- Find the similarity between documents  
- Submit a query and return the most relevant documents
- Able to find the (dis)similarity with or without words in common
- Uses word2vec vector embeddings of words
- Uses bag-of-words representation of the documents
- Find the traveling distance between documents:
  the most efficient way to “move” the distribution of document 1 to the distribution of document 2
- Return: top 5 candidate answer texts with their similarity score traversing the article with the query
  [1]: “From Word Embeddings To Document Distances” by Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, and Kilian Q. Weinberger

2. **Question Classification**  

- Wh- Questions

> Q starts with "who" or Q contains "whom" that is not followed by a comma: [PERSON]
> Q starts with "where": [LOCATION]
> Q starts with "when": [DATE]
> Q stars with "what": [OBJECT]

- **Binary Questions**

> Q starts with  be verbs in any tense, regardless contraction or negation
> E.g. is, was, weren’t
> Q starts with can/have/will in any tense, regardless contraction or negation
> E.g.  could, couldn’t, would

- Other: descriptive / reasoning questions

3. **Answer Type - [PERSON]**

1) Look for noun phrases with the NER tags: [PERSON] / [NORP] / [ORG] in the text that contains answer.   
2) Look for noun phrases where proper noun is the subject  within sentence.  
3) If 2 is not satisfied, look for NER tags with [PERSON] / [ORG] / [NORP] .  
4) If in 3 there are more than one phrase with the specified NER tags, apply the **novelty rule** . 

> Novelty rule: answer = the phrase in the candidate answer is novel, that is, not in the query  

5) If in 4 more than one NER tags are novel, do the dependency parsing and look for the one with the closest distance from the candidate text’s ROOT . 
6) Formalize the Answer . 
*Default*: the candidate text with the highest similarity score traversing the article with the query

4. **Answer Type - [LOCATION]**

1)Look for noun phrases with the NER tags: [GPE] / [LOC], in the candidate text . 
2)If more than one correct NER in the candidate text, apply the novelty rule . 

> Text = "The Old Kingdom is perhaps best known for the large number of pyramids constructed at this time as burial places for Egypt's kings" .  
> Q = "What is perhaps best known for the large number of pyramids constructed at this time as burial places for Egypt's kings?"   
> Phrases with NER tags in candidate text: {'Old Kingdom': 'GPE', 'Egypt': 'GPE'} .  
> **Novelty rule**: ans should be the one that doesn't appear in the Q: "Old Kingdom" .  
> 3)If in 2 more than one noun phrases with correct NER tags are novel, do the dependency parsing and look for the one with the closest distance from the candidate text’s ROOT   
> 4)Formalize the answer . 
> Default: the candidate text with the highest similarity score traversing the article with the query

5. **Answer Type - [OBJECT]**  
   1) Restructure the Query and find the similar pattern in the candidate texts

   2) If 1 is met and candidate text has similar pattern as the query, find NER tags in the text before the position where similar pattern was found, **novelty rule** still holds

   > Text: “**Pittsburgh** was *named* in 1758 by General **John Forbes**, in honor of British statesman **William Pitt**, 1st Earl of **Chatham**.”
   >
   > Q: “What was named in 1758 by General **John Forbes**?”
   >
   > - NER: bold; ROOT: italic; Query Pattern: underline

   3) If in 2 more than one candidate noun phrases are novel, do the dependency parsing and find the one closest to the text’s ROOT

   Default: if in 2 the similar pattern is found in candidate text but there are no NER tags found, reconstruct the text to find the answer

6. **WH- Question Answers Summary**

   1) Find correct NER noun phrasesIf more than one correct NER, apply the novelty rule

   2) If the novelty rule fail because there are more than one correct NER phrases that do not appear in the query, do the dependency parsing of the text and find the novel NER that is closest to the ROOT of the text

   3) Formalize the answer
   4) If all steps failed, go for the default answer

   This is the generic logical flow but specific rules apply for different answer type under WH-question -- e.g.[PERSON]: first looks for person the  related NER phrase  that is  also the subject within the text

7. **Answer Type - [BINARY]**

   Do the sentiment analysis for the candidate text with the highest similarity score traversing the article with the query

   If the polarity score is greater than - 0.1, the answer is “Yes”, and “No” otherwise.- 0.1 is chosen because after some test using the pre-trained model, - 0.1 as the threshold yields a more accurate result than using 0 as the threshold

   Default: “Yes.”

8. **Answer Type - [DESCRIPTION]/[REASON]**

   Find the top candidate answer text with high similarity score traversing the article with the query

   If the question ask for describing certain event in a high level, then the top candidate answer text with the highest similarity score traversing the article with the query is very likely to give a relevant information the query wants.

   Restructure the candidate answer text to make it grammarly correct







#### Collaborators:

[@ernestomv26](https://github.com/ernestomv26/)
[@ericgeyiyang](https://github.com/ericgeyiyang/)
[@surancy](https://github.com/surancy/)
Chaitanya Dwivedi
