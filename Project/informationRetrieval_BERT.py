from tqdm import tqdm
import torch
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from nltk import flatten 

class InformationRetrieval():

        def __init__(self):
            
            # We will be using 'paraphrase-MiniLM-L6-v2' from Hugging Face
            self.index = None
            self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        def buildIndex(self, docs, docIDs):
            
            # Convert each doc into a paragraph
            sent_docs = []
            for doc in docs:
                sent_doc = ""
                for sentence in doc:
                    for word in sentence:
                        sent_doc = sent_doc + word + " "
                sent_docs.append(sent_docs)
            
            print("Number of documents :", len(sent_docs))

            self.docIDs = docIDs

            # Compute Document Embeddings (if not done already)
            index = []
            if os.path.exists('index.pt') == False:
                with torch.no_grad():
                    for text in tqdm(sent_docs):
                        embedding = self.model.encode(text)
                        index.append(embedding)
                torch.save(index, 'index.pt')
            else:
                index = torch.load('index.pt')
            
            print("Shape of document embeddings, index : ", len(index), len(index[0]))
            self.index = index
        
        def cosine_similarity_vecs(self, query, doc):
            
            # Given two vectors, returns the cosine similarity between the two
            if np.linalg.norm(query) == 0 or np.linalg.norm(doc) == 0:
                return 0
            else:
                return np.dot(query, doc)/(np.linalg.norm(query)*np.linalg.norm(doc))
            

        def rank(self, queries):
            
            # Convert each query into a paragraph
            sent_queries = []
            for query in queries:
                sent_query = ""
                for sentence in query:
                    for word in sentence:
                        sent_query = sent_query + word + " "
                sent_queries.append(sent_query)

            print("Number of Queries:", len(sent_queries))

            # Final Ranking Order
            doc_IDs_ordered = []

            # Compute Query Embeddings (if not done already)
            query_embs = []
            if os.path.exists('queries.pt') == False:
                with torch.no_grad():
                    for text in tqdm(sent_queries):
                        embedding = self.model.encode(text)
                        query_embs.append(embedding)
                torch.save(query_embs, 'queries.pt')

            else:
                query_embs = torch.load('queries.pt')

            # Compute Ranking Order of docs for each query
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