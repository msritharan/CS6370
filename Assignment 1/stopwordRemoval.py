from util import *

# Add your import statements here
from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english'))


class StopwordRemoval():

	def fromList(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
			representing a sentence with stopwords removed
		"""

		stopwordRemovedText = None

		#Fill in code here
		stopwordRemovedText = []
		for sentence in text:
			modified_sentence = []
			for word in sentence:
				if word in stop_words:
					continue
				else:
					modified_sentence.append(word)
			stopwordRemovedText.append(modified_sentence)

		return stopwordRemovedText




	