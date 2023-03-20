from util import *

# Add your import statements here
import numpy as np



class InformationRetrieval():

	def __init__(self):
		self.index = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = {}

		#Fill in code here

		# find document by id
		doc_by_id = {}
		for idx in range(len(docIDs)):
			doc_by_id[docIDs[idx]] = docs[idx]

		# print(doc_by_id)

		# find term by idx and idx by term
		index_by_term = {}
		term_by_index = {}
		idx = 0
		for doc in docs:
			for sentence in doc:
				for word in sentence:
					if word not in index_by_term:
						index_by_term[word] = idx
						term_by_index[idx] = word
						idx += 1
		
		# print(term_by_index)
		# print(index_by_term)

		# term frequency of docs
		term_freq = {}
		for id in docIDs:
			f = np.zeros(len(term_by_index))
			for sentence in doc_by_id[id]:	
				for word in sentence:
					f[index_by_term[word]] += 1
			term_freq[id] = f

		# print(term_freq)

		# document frequency of terms
		doc_freq = np.zeros(len(term_by_index))
		for doc_id in docIDs:
			for term_id in range(len(term_freq[doc_id])):
				if term_freq[doc_id][term_id] > 0:
					doc_freq[term_id] += 1
		
		# print(doc_freq)

		# compute vector representation of each doc using normalized tf-idf
		for doc_id in docIDs:
			doc_rep = term_freq[doc_id]/np.max(term_freq[doc_id])
			doc_rep *= np.log(len(docIDs)/doc_freq)
			index[doc_id] = doc_rep

		self.term_by_index = term_by_index
		self.index_by_term = index_by_term
		self.doc_by_id = doc_by_id
		self.doc_freq = doc_freq
		self.term_freq = term_freq
		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here
		for query in queries:
			# find query rep
			query_rep = np.zeros(len(self.index_by_term))
			query_f = np.zeros(len(self.index_by_term))
			for sentence in query:
				for word in sentence:
					if word in self.index_by_term:
						query_f[self.index_by_term[word]] += 1
			
			# print(query_f)
			query_rep = 0.5*query_f/np.max(query_f) + 0.5*(query_f > 0)
			query_rep *= np.log(len(self.doc_by_id)/self.doc_freq)
			# print(query_rep)

			# find cosine similarity ordering
			doc_score = []
			for doc_id in self.index:
				doc_rep = self.index[doc_id]
				score = np.dot(query_rep, doc_rep)/(np.linalg.norm(query_rep)*np.linalg.norm(doc_rep))
				doc_score.append([score, doc_id])
			doc_score = np.array(doc_score)
			doc_score = doc_score[-doc_score[:,0].argsort()]

			# print(doc_score)

			ret_doc = [x[1] for x in doc_score]
			doc_IDs_ordered.append(ret_doc)

		return doc_IDs_ordered




