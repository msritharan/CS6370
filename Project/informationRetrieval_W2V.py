import itertools
import numpy as np
from gensim import downloader
import os
from tqdm import tqdm

class InformationRetrieval():
    
    def __init__(self):
        
            self.index = None
        
    
    def buildIndex(self, docs, docIDs):
        
            # Index will contain document embeddings of the given training dataset
            index = []
            
            # Download Pre-trained model by Hugging Face
            self.model = downloader.load('word2vec-google-news-300')
            self.vocab = self.model.index_to_key
            self.docIDs = docIDs

            # Computation of document embeddings (if not done already)
            if os.path.exists('word2vec_google.npy') == False:
                for doc in tqdm(docs):
                    doc_count = 0
                    doc_emb = np.zeros(300)
                    if len(doc) == 0:
                        continue
                    for sentence in doc:
                        for word in sentence:
                            if word in self.vocab:
                                doc_emb += self.model[word]
                                doc_count += 1
                    doc_emb /= doc_count
                    index.append(doc_emb)
                np.save('word2vec_google.npy', index)
            else:
                index = np.load('word2vec_google.npy')
            
            # Document Embeddings
            self.index = index

    def cosine_similarity_vecs(self, query, doc):
            
            # Given two vectors, returns the cosine similarity between the two
            if np.linalg.norm(query) == 0 or np.linalg.norm(doc) == 0:
                return 0
            else:
                return np.dot(query, doc)/(np.linalg.norm(query)*np.linalg.norm(doc))
    
    
    def rank(self, queries):

            # Contains the ranking order for each query
            doc_IDs_ordered = []
            
            # construct easily iterable queries list
            iter_queries = []
            for i in queries:
                q = list(itertools.chain.from_iterable(i))
                iter_queries.append(q)

            # Computation of Query Embeddings
            query_embs = []
            for q in tqdm(iter_queries):
                q_count = 0
                q_emb = np.zeros(300)
                for word in q:
                    if word in self.vocab:
                        q_emb += self.model[word]
                        q_count += 1
                if q_count != 0:
                    q_emb /= q_count
                query_embs.append(q_emb)

            # Computation of Ranking Order
            Q = len(query_embs)
            D = len(self.index)
            cos_sim_mat = np.zeros((Q, D))
            for q_id in range(Q):
                for d_id in range(D):
                    cos_sim_mat[q_id][d_id] = self.cosine_similarity_vecs(query_embs[q_id], self.index[d_id])

            for q_id in range(Q):
                doc_scores = cos_sim_mat[q_id]
                sorted_doc_idxs = np.flip(doc_scores.argsort())
                sorted_doc_ids = [self.docIDs[idx] for idx in sorted_doc_idxs]
                doc_IDs_ordered.append(sorted_doc_ids)

            return doc_IDs_ordered