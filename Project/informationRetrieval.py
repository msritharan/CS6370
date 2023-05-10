from util import *

# Add your import statements here
import numpy as np



class InformationRetrieval():

	def __init__(self):
		self.index = {}

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

		# index will be of the form : {"doc_id" : {"term" : tf_idf value}}
		self.index = {}

		# Computation of tf values
		self.tf = {}	# form : {"doc_id" : {"term" : tf}}
		num_docs = len(docIDs)
		for idx in range(num_docs):
			doc_id = docIDs[idx]
			self.tf[doc_id] = {}
			total_words = 0
			for sentence in docs[idx]:
				for word in sentence:
					if word in self.tf[doc_id]:
						self.tf[doc_id][word] += 1
					else:
						self.tf[doc_id][word] = 1
					total_words += 1
			
			# normalize tf value
			# if total_words != 0:
			# 	for term in self.tf[doc_id]:
			# 		self.tf[doc_id][term] /= total_words
			

		# Computation of idf values
		self.idf = {}	# form : {"term" : idf}
		# computes df values
		for doc_id in docIDs:
			for term in self.tf[doc_id]:
				if term in self.idf:
					self.idf[term] += 1
				else:
					self.idf[term] = 1
		# converts to idf smooth values
		for term in self.idf:
			self.idf[term] = np.log(num_docs/(self.idf[term]))

		# Computation of tf_idf values
		for doc_id in docIDs:
			self.index[doc_id] = {}
			for term in self.tf[doc_id]:
				self.index[doc_id][term] = self.tf[doc_id][term]*self.idf[term]

	def cosine_similarity(self, query, doc):
		# Compute magnitude of query
		mag_query = 0
		for term in query:
			mag_query += (query[term])**2
		mag_query = np.sqrt(mag_query)

		# Compute magnitude of do
		mag_doc = 0
		for term in doc:
			mag_doc += (doc[term])**2
		mag_doc = np.sqrt(mag_doc)

		if mag_doc == 0 or mag_query == 0:
			return 0
		
		# Dot Product of query and doc
		dot_prod = 0
		for term in query:
			if term in doc:
				dot_prod += query[term]*doc[term]
		
		# Cosine Similarity
		cos_sim = dot_prod/(mag_query*mag_doc)

		return cos_sim
	
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

		for query in queries:
			# query representation
			query_rep = {}
			# tf value
			total_words = 0
			for sentence in query:
				for word in sentence:
					if word not in self.idf:
						continue
					if word in query_rep:
						query_rep[word] += 1
					else:
						query_rep[word] = 1
				total_words += 1
			# normalize tf and mul by idf
			for term in query_rep:
				# query_rep[term] /= total_words
				query_rep[term] *= self.idf[term]
			
			# compute cos sim scores
			doc_scores = {}
			for doc_id in self.index:
				cos_sim_score = self.cosine_similarity(query_rep, self.index[doc_id])
				doc_scores[doc_id] = cos_sim_score
			doc_scores = sorted(doc_scores.items(), key=lambda x:x[1], reverse= True)
			
			# order of the retrieved documents
			doc_order = []
			for doc in doc_scores:
				doc_order.append(doc[0])

			doc_IDs_ordered.append(doc_order)

		return doc_IDs_ordered


