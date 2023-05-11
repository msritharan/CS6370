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
            index = {}
            doc_id2index = {}
            index2doc_id = {}
            

            # Computation of tf values
            tf = {}	# form : {"doc_id" : {"term" : tf}}
            tokens = []
            term2index = {}
            index2term = {}
            
            num_docs = len(docIDs)
            for idx in range(num_docs):
                doc_id = docIDs[idx]
                doc_id2index[doc_id] = idx
                index2doc_id[idx] = doc_id

                tf[doc_id] = {}
                total_words = 0
                for sentence in docs[idx]:
                    for word in sentence:
                
                        if word in tf[doc_id]:
                            tf[doc_id][word] += 1
                        else:
                            tf[doc_id][word] = 1
                        total_words += 1

                        if word not in tokens:
                            tokens.append(word)
            
            # Computation of idf values
            idf = {}	# form : {"term" : idf}

            # computes df values
            for doc_id in docIDs:
                for term in tf[doc_id]:
                    if term in idf:
                        idf[term] += 1
                    else:
                        idf[term] = 1

            # converts to idf smooth values
            for term in idf:
                idf[term] = np.log(num_docs/(idf[term]))

            # Computation of tf_idf values
            for doc_id in docIDs:
                index[doc_id] = {}
                for term in tf[doc_id]:
                    index[doc_id][term] = tf[doc_id][term]*idf[term]

            for idx in range(len(tokens)):
                term2index[tokens[idx]] = idx
                index2term[idx] = tokens[idx]
            
            # LSA
            # Construction of Term Document matrix
            nDocs = len(docIDs)
            nTerms = len(tokens)
            td_mat = np.zeros((nTerms, nDocs))
            for i in range(nTerms):
                for j in range(nDocs):
                    if index2term[i] in index[index2doc_id[j]]:
                        td_mat[i, j] = index[index2doc_id[j]][index2term[i]]

            # Perform SVD
            U, Singluar_Values, V = np.linalg.svd(td_mat)
            S = np.diag(Singluar_Values)
            
            # Choosing the value of K
            # total_singular_val = np.sum(Singluar_Values)
            # print("Total sum of singular values = ", total_singular_val)
            # k = 0
            # sum_k_singular_val = 0
            # while k < S.shape[0]:
            #     sum_k_singular_val += S[k, k]
            #     if sum_k_singular_val >= 0.8*total_singular_val:
            #         break
            #     k += 1
            # 

            k = 250
            print("Value of k = ", k)
            
            Uk = U[:, :k]
            Sk = S[:k,:k]
            Vk = V[:k, :]

            td_mat_lsa = np.linalg.inv(Sk)@Uk.T@td_mat

            self.tf = tf
            self.idf = idf
            self.index = index
            self.td_mat = td_mat
            self.index2term = index2term
            self.index2doc_id = index2doc_id
            self.doc_id2index = doc_id2index
            self.term2index = term2index
            self.docIDs = docIDs
            self.nDocs = nDocs
            self.nTerms = nTerms
            self.Uk = Uk
            self.Sk = Sk
            self.Vk = Vk
            self.td_mat_lsa = td_mat_lsa

        def cosine_similarity_vecs(self, query, doc):
            if np.linalg.norm(query) == 0 or np.linalg.norm(doc) == 0:
                return 0
            else:
                return np.dot(query, doc)/(np.linalg.norm(query)*np.linalg.norm(doc))

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
            
            ranks = []

            for query in queries:
                
                # query representation
                query_rep = np.zeros(self.nTerms)
                
                # tf value
                total_words = 0
                for sentence in query:
                    for word in sentence:
                        
                        if word not in self.idf:
                            continue
                        
                        query_rep[self.term2index[word]] += 1
                    
                    total_words += 1
                
                # normalize tf and mul by idf
                for idx in range(len(query_rep)):
                    # query_rep[term] /= total_words
                    query_rep[idx] *= self.idf[self.index2term[idx]]
                
                # compute cos sim scores
                vq = np.linalg.inv(self.Sk) @ self.Uk.T @ query_rep
    
                scores = []
                for i in range(self.nDocs):
                    vd = self.td_mat_lsa[:, i]
                    scores.append([self.cosine_similarity_vecs(vq, vd), self.docIDs[i]])
                
                scores.sort(reverse=True)
                ranks.append([scores[d_ind][1] for d_ind in range(len(scores))])

            docIDsOrdered = ranks
            print(np.array(docIDsOrdered).shape)
            return docIDsOrdered


