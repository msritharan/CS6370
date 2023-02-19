# from util import *

# Add your import statements here
from nltk.stem import WordNetLemmatizer



class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here
		reducedText = []
		wordnet_lemmatizer = WordNetLemmatizer()
		for sentence in text:
			reduced_sentence = []
			for word in sentence:
				reduced_sentence.append(wordnet_lemmatizer.lemmatize(word))
			reducedText.append(reduced_sentence)
		
		return reducedText


