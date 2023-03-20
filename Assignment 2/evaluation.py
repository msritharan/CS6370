from util import *

# Add your import statements here
import numpy as np


class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		relevant_docs = 0
		for idx in range(min(k, len(query_doc_IDs_ordered))):
			doc  = query_doc_IDs_ordered[idx]
			if doc in true_doc_IDs:
				relevant_docs += 1
		precision = relevant_docs/k

		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		meanPrecision = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if qrel['query_num'] == query_id:
					query_true_doc_IDs.append(qrel['id'])
			
			# precision for a particular query
			query_precision = self.queryPrecision(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanPrecision += query_precision
		
		meanPrecision /= num_queries

		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		relevant_docs = 0
		for idx in range(min(k, len(query_doc_IDs_ordered))):
			doc  = query_doc_IDs_ordered[idx]
			if doc in true_doc_IDs:
				relevant_docs += 1
		recall = relevant_docs/len(true_doc_IDs)

		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		meanRecall = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if qrel['query_num'] == query_id:
					query_true_doc_IDs.append(qrel['id'])
			
			# recall for a particular query
			query_recall = self.queryRecall(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanRecall += query_recall
		
		meanRecall /= num_queries

		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here
		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k) 

		fscore = 2*precision*recall/(precision + recall)

		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		meanFscore = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if qrel['query_num'] == query_id:
					query_true_doc_IDs.append(qrel['id'])
			
			# recall for a particular query
			query_fscore = self.queryFscore(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			meanFscore += query_fscore
		
		meanFscore /= num_queries

		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4: list of dicts (added 4th arg since we need rel scores)
		arg5 : int
			The k value
		

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		# Find DCG
		DCG = 0.0
		for i in range(1, min(k, len(query_doc_IDs_ordered)) + 1):
			# find relevance of document with query
			rel = 0
			for qrel in qrels:
				if qrel['query_num'] == query_id and qrel['id'] == query_doc_IDs_ordered[i - 1]:
					rel = 5 - qrel['position']
			DCG += rel/np.log2(i + 1)

		# Find set of relevant scores for true docs
		true_rel = []
		for i in range(0, len(true_doc_IDs)):
			# find relevance of document with query
			rel = 0
			for qrel in qrels:
				if qrel['query_num'] == query_id and qrel['id'] == true_doc_IDs[i]:
					rel = 5 - qrel['position']
			true_rel.append(rel)

		true_rel = np.array(true_rel, dtype= int)
		true_rel = -np.sort(-true_rel)
		
		#Find IDCG
		IDCG = 0.0
		for i in range(0, min(k, len(query_doc_IDs_ordered))):
			rel = true_rel[i]
			IDCG += rel/np.log2(i + 1)

		# Find nDCG
		nDCG = DCG/IDCG

		return nDCG


	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		meanNDCG = -1

		#Fill in code here
		meanNDCG = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if qrel['query_num'] == query_id:
					query_true_doc_IDs.append(qrel['id'])
			
			# recall for a particular query
			query_NDCG = self.queryNDCG(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, qrels, k)
			meanNDCG += query_NDCG
		
		meanNDCG /= num_queries

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		avgPrecision = 0
		for idx in range(min(k, len(query_doc_IDs_ordered))):
			doc = query_doc_IDs_ordered[idx]
			if doc in true_doc_IDs:
				avgPrecision += self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, idx + 1)

		avgPrecision /= len(true_doc_IDs)

		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		meanAveragePrecision = 0
		num_queries = len(query_ids)
		for idx in range(num_queries):
			# for each query id, we would want to find the true_doc_IDs
			query_id = query_ids[idx]
			query_true_doc_IDs = []
			for qrel in qrels:
				if qrel['query_num'] == query_id:
					query_true_doc_IDs.append(qrel['id'])
			
			# compute average precision
			meanAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[idx], query_id, query_true_doc_IDs, k)
			
		meanAveragePrecision /= num_queries

		return meanAveragePrecision

