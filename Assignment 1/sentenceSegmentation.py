from util import *

# Add your import statements here
import nltk
sentence_delimiters = ".!?"

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		segmentedText = []
		sentence = ""
		for character in text:
			if character in sentence_delimiters:
				segmentedText.append(sentence)
				sentence = ""
			else:
				sentence += character

		return segmentedText


	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here
		tokenizer = nltk.data.load('nltk:tokenizers/punkt/english.pickle')
		segmentedText = tokenizer.tokenize(text)
		
		return segmentedText