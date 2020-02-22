# JSONtoTXT.py 
# Reads news articles from a JSON file
# Splits the content into sentences
# Cleans and normalizes the content
# Write each processed sentence into a text file


import json
import spacy
from spacy.lang.en import English # updated
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
import re

# Loads the spaCy small English language model
nlp = spacy.load('en_core_web_sm')

# Removes stopwords from spaCy default stopword list
nlp.Defaults.stop_words -= {"my_stopword_1", "my_stopword_2"}

# Adds custom stopword into spaCy default stopword list
nlp.Defaults.stop_words |= {"my_stopword_1", "my_stopword_2"}

print(nlp.Defaults)

# Calculates the frequency of words in a document
def word_frequency(my_doc):

	# all tokens that arent stop words or punctuations
	words = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True]

	# noun tokens that arent stop words or punctuations
	nouns = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]

	# verb tokens that arent stop words or punctuations
	verbs = [token.text for token in my_doc if token.is_stop != True and token.is_punct != True and token.pos_ == "VERB"]

	# five most common tokens
	word_freq = Counter(words)
	common_words = word_freq.most_common(5)
	print("---------------------------------------")
	print("5 MOST COMMON TOKEN")
	print(common_words)
	print("---------------------------------------")
	print("---------------------------------------")

	# five most common noun tokens
	noun_freq = Counter(nouns)
	common_nouns = noun_freq.most_common(5)
	print("5 MOST COMMON NOUN")
	print(common_nouns)
	print("---------------------------------------")
	print("---------------------------------------")

	# five most common verbs tokens
	verb_freq = Counter(verbs)
	common_verbs = verb_freq.most_common(5)
	print("5 MOST COMMON VERB")
	print(common_verbs)
	print("---------------------------------------")
	print("---------------------------------------")

def remove_stopwords(sentence):
	sentence = nlp(sentence)
	processed_sentence = ' '.join([token.text for token in sentence if token.is_stop != True ])
	return processed_sentence

def remove_punctuation_special_chars(sentence):
	sentence = nlp(sentence)
	processed_sentence = ' '.join([token.text for token in sentence 
		if token.is_punct != True and 
		   token.is_quote != True and 
		   token.is_bracket != True and 
		   token.is_currency != True and 
		   token.is_digit != True])
	return processed_sentence

def lemmatize_text(sentence):
    sentence = nlp(sentence)
    processed_sentence = ' '.join([word.lemma_ for word in sentence])
    return processed_sentence

def remove_special_chars(text):
	bad_chars = ["%", "#", '"', "*"] 
	for i in bad_chars: 
		text = text.replace(i, '')
	return text

def split_sentences(document):
	sentences = [sent.string.strip() for sent in doc.sents]
	return sentences

sentence_index = 0	

with open('/Users/erdemisbilen/TFvenv/articles.json') as json_file:
	data = json.load(json_file)
		
	with open("article_all.txt", "w") as text_file:
		for p in data:
			article_body = p['article_body']
			article_body = remove_special_chars(article_body)

			doc = nlp(article_body)

			sentences = split_sentences(doc)
			word_frequency(doc)

			for sentence in sentences:
				sentence_index +=1
				print("Sentence #" + str(sentence_index) + "----------------------------------------------")
				print("Original Sentence               : " + sentence)
				sentence = remove_stopwords(sentence)
				sentence = remove_punctuation_special_chars(sentence)
				print("Stopwors and Punctuation Removal: " + sentence)
				sentence = lemmatize_text(sentence)
				print("Lemmitization Applied           : " + sentence)
				text_file.write(sentence + '\n')
		
	text_file.close()
