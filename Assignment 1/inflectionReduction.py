from util import *

# Add your import statements here
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

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
		for sentence in text:
			words = []
			for word in sentence:
				stemmed_word = stemmer.stem(word)
				words.append(stemmed_word)
			reducedText.append(words)

		return reducedText


