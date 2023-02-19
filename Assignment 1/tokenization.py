# from util import *

# Add your import statements here
from nltk.tokenize.treebank import TreebankWordTokenizer



class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		tokenizedText = []
		for sentence in text:
			words = []
			word = ""
			for character in sentence:
				if(character == ' '):
					words.append(word)
					word = ""
				else:
					word += character
			tokenizedText.append(words)

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		tokenizedText = []
		for sentence in text:
			tokens = TreebankWordTokenizer().tokenize(sentence)
			tokenizedText.append(tokens)

		return tokenizedText