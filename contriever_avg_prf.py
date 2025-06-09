"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

import os
import ast
import argparse
import pandas as pd 
import numpy as np
from tqdm import tqdm

from pyserini.search.lucene import LuceneSearcher
from pyserini.encode import AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from pyserini.search.faiss import FaissSearcher 

import pandas as pd
from index_paths import THE_TOPICS, THE_SPARSE_INDEX, THE_DENSE_INDEX, load_queries_qids

import json 
from numpy import dot
from numpy.linalg import norm
from rede_rf import REDE_RF
from prompter import Promptor
from generator import MistralGenerator
from transformers.utils import logging

logging.set_verbosity_error() 

def load_document_vector(doc_id, searcher):
    faiss_doc_index = [i for i in range(len(searcher.docids)) if searcher.docids[i] == doc_id][0]
    return searcher.index.reconstruct(faiss_doc_index)

class REDE_PRF(REDE_RF):
    def __init__(self, 
                prompter,
                encoder, 
                sparse_searcher, 
                faiss_searcher, 
                init_retrieval_method='hybrid'):

        self.prompter = prompter
        self.encoder = encoder
        self.sparse_searcher = sparse_searcher
        self.faiss_searcher = faiss_searcher
        self.init_retrieval_method = init_retrieval_method

        if self.init_retrieval_method == 'hybrid':
            assert(self.faiss_searcher is not None)
            self.hybrid_searcher = HybridSearcher(self.faiss_searcher, self.sparse_searcher )

        self.docid_dict = {self.faiss_searcher.docids[i]: i for i in range(len(self.faiss_searcher.docids))}

    def e2e_search(self, query, k_init=20, k=10, prf_depth=3):
        # Run initial retrieval pass
        retrieved_documents = self.initial_retrieval(query, k_init)
        relevant_docs, is_hypothetical_docs = [docid for passage, docid, search_type in retrieved_documents], False   
        assert(len(relevant_docs) == k_init)
        relevant_docs = relevant_docs[:prf_depth]
        assert(len(relevant_docs) == prf_depth)
        query_vector = self.encoder.encode([query])
        rede_vector = self.encode_rede_vector(query_vector, relevant_docs, is_hypothetical_docs)
        rede_hits = self.faiss_searcher.search(rede_vector, k=k)
        return rede_hits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate ReDE-PRF')
    parser.add_argument('--query_encoder_path', default='facebook/contriever', help='query encoder path')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate rede-rf')
    parser.add_argument('--init_retrieval_method', default='hybrid', help='search method for initial retrieval')
    parser.add_argument('--k_init', type=int, default=20, help='number of documents for the LLM-Rf to judge')
    parser.add_argument('--k', type=int, default=10, help='number of documents to return for final retrieval')
    parser.add_argument('--prf_depth', type=int, default=20, help='number of documents to return for final retrieval')
    parser.add_argument('--output_filename', required=True,  help='output filename')
    args = parser.parse_args()

    qids, queries = load_queries_qids(args.corpus_name)
    # Load searchers
    lucence_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    encoder = AutoQueryEncoder(encoder_dir=args.query_encoder_path, pooling='mean')
    if args.corpus_name == 'dl19' or args.corpus_name == 'dl20':
        faiss_searcher = FaissSearcher(THE_DENSE_INDEX[args.corpus_name], encoder)
    else:
        faiss_searcher = FaissSearcher.from_prebuilt_index(THE_DENSE_INDEX[args.corpus_name], encoder)

    ###############################
    # Load prompter
    # Prompter not needed, but used to help extract document text
    prompter = Promptor(task=args.corpus_name)

    rede_rf = REDE_PRF(prompter=prompter,
                       encoder=encoder, 
                       sparse_searcher=lucence_searcher, 
                       faiss_searcher=faiss_searcher, 
                       init_retrieval_method=args.init_retrieval_method)

    output_filename = f'{args.output_filename}_rede-prf_{args.init_retrieval_method}_k_init-{args.k_init}_prf_depth-{args.prf_depth}'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = rede_rf.e2e_search(query, k_init=args.k_init, k=args.k, prf_depth=args.prf_depth)
            rank = 0
            for hit in hits:
                rank += 1
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')

    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))