# TEXT-CLASSIFICATION-FOR-TELUGU-USING-NATURAL-LANGUAGE-PROCESSING
With the exponential growth of digital content in the Telugu language, the 
need for automated news classification has become paramount. This project focuses 
on harnessing the power of Natural Language Processing (NLP) to create an efficient 
system for classifying Telugu news articles into relevant categories. Our approach 
involves preprocessing Telugu text, feature extraction, and the application of 
advanced NLP models and techniques. We have compiled a diverse dataset of Telugu 
news articles from various sources. Through rigorous experimentation and 
evaluation, we demonstrate the efficacy of NLP-driven classification, which promises 
to revolutionize content organization and user experience in the Telugu news domain. 
This project showcases the potential of NLP in automating and improving Telugu 
news categorization, ensuring that readers can access the information that matters 
most to them.

## main.py
```python
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
import seaborn as sns
import os
telugu_news_df = pd.read_csv("/train_telugu_news.csv")
telugu_news_df
telugu_news_df.isna().sum()
telugu_news_df[telugu_news_df["heading"].isna() == True]
del telugu_news_df["heading"]
del telugu_news_df["SNo"]
telugu_news_df["topic"].unique()
topic_dic = {}
c = 0
for un in telugu_news_df["topic"].unique():
 if un not in topic_dic:
 topic_dic[un] = c
 c += 1
topic_dic
invtopic_dict = {v: k for k, v in topic_dic.items()}
def func_topic(s):
 return topic_dic[s]
telugu_news_df["topic"] = telugu_news_df["topic"].apply(func_topic)
date_df = telugu_news_df["date"]
del telugu_news_df["date"]
pip install indic-nlp-library
from indicnlp.tokenize import sentence_tokenize
indic_string = telugu_news_df["body"][0]
# Split the sentence, language code "hi" is passed for hingi
sentences=sentence_tokenize.sentence_split(indic_string, lang='te')
# print the sentences
for t in sentences:
 print(t)
telugu_news_df["body_processed"] = telugu_news_df["body"].str.replace('\u200c', '')
telugu_news_df["body_processed"] = telugu_news_df["body_processed"].str.replace('\n', '')
telugu_news_df["body_processed"] = telugu_news_df["body_processed"].str.replace('\t', '')
telugu_news_df["body_processed"] = telugu_news_df["body_processed"].str.replace('\xa0',
'')
PUNCT = string.punctuation
def remove_punctuation(text):
 return text.translate(str.maketrans('', '', PUNCT))
telugu_news_df["body_processed"][6665]
from indicnlp.tokenize import sentence_tokenize
tot_telugu_text1 = ""
for t in telugu_news_df["body_processed"]:
 tot_telugu_text1 += t
tot_sentences = sentence_tokenize.sentence_split(tot_telugu_text1, lang='te')
print(len(tot_sentences))
telugu_news_df["body_processed"] = telugu_news_df["body_processed"].apply(lambda text:
remove_punctuation(text))
del telugu_news_df["body"]
tot_telugu_text = ""
c = 1
for t in telugu_news_df["body_processed"]:
 tot_telugu_text += t
 c += 1
print(c)
len(tot_telugu_text)
from indicnlp.tokenize import indic_tokenize
vocab_dic = {}
tokenized_text = []
heap_arr = []
for t in indic_tokenize.trivial_tokenize(tot_telugu_text):
 tokenized_text.append(t)
 heap_arr.append(len(vocab_dic))
 if t not in vocab_dic:
 vocab_dic[t] = 1
 else:
 vocab_dic[t] += 1
from nltk.probability import FreqDist
freq_dist = FreqDist(vocab_dic)
vocab_dic_sorted = {k: v for k, v in sorted(vocab_dic.items(), key=lambda item: item[1],
reverse = True)}
 top_k_words.append([key, vocab_dic_sorted[key]])
 c += 1
top_k_words_df = pd.DataFrame(top_k_words)
top_k_words_df.columns = ["word", "freq"]
top_k_words_df.head()
telugu_words = list(vocab_dic_sorted.keys())
tot_sentences_proc = []
bigrams_telugu_vocab = {k: v for k, v in sorted(bigrams_telugu_vocab.items(), key=lambda
item: item[1], reverse = True)}
print("Total no.of unique bi-grams :- ", len(bigrams_telugu_vocab))
k = 30
print("Top" ,k ,"most-occuring bi-grams in the corpus are\n")
c = 0
for key in bigrams_telugu_vocab:
 if c == k:
 break
 print(key , " -> ", bigrams_telugu_vocab[key])
 c += 1
four_grams_telugu_vocab = {}
five_grams_telugu_vocab = {k: v for k, v in sorted(five_grams_telugu_vocab.items(),
key=lambda item: item[1], reverse = True)}
k = 30
print("Top" ,k ,"most-occuring 5-grams in the corpus are\n")
c = 0
for key in five_grams_telugu_vocab:
 if c == k:
 break
 print(key , " -> ", five_grams_telugu_vocab[key])
 c += 1
ngrams_count = []
ngrams_count.append(len(vocab_dic_sorted))
ngrams_count.append(len(bigrams_telugu_vocab))
ngrams_count.append(len(four_grams_telugu_vocab))
ngrams_count.append(len(five_grams_telugu_vocab))
ngrams_count
plt.figure(figsize = (10,5))
plt.bar(x = ["1-gram","2-gram","4-gram","5-gram"], height = ngrams_count)
plt.ylabel("No.of unique n-grams")
```

## OUTPUT
![WhatsApp Image 2024-02-15 at 1 27 51 PM](https://github.com/veerapallijanith/TEXT-CLASSIFICATION-FOR-TELUGU-USING-NATURAL-LANGUAGE-PROCESSING/assets/75234814/9f78a6a2-86e9-4c50-8b83-a6b7dc271401)

