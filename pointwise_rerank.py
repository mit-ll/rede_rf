"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

import os
import sys
import ast
import argparse
import pickle
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
from pyserini.search.lucene import LuceneSearcher
from rf import LLM_RF
from prompter import Promptor
from generator import MistralGenerator, GemmaGenerator, Llama3Generator
from transformers.utils import logging

logging.set_verbosity_error() 

class PointwiseRerank(LLM_RF):
    def __init__(self, 
                 prompter, 
                 encoder,
                 sparse_searcher, 
                 faiss_searcher, 
                 instruction_llm, 
                 init_retrieval_method='hybrid'):

        super().__init__(prompter, instruction_llm, encoder, faiss_searcher, sparse_searcher, init_retrieval_method)

    def e2e_search(self, query, k_init):
        retrieved_documents = self.initial_retrieval(query, k_init)
        _, _, llm_reranking_results = self.judge_documents(query, retrieved_documents)
        return llm_reranking_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Pointwise Reranking')
    parser.add_argument('--instruction_model_path', help='instruction model path')
    parser.add_argument('--query_encoder_path', default='facebook/contriever', help='query encoder path')
    parser.add_argument('--prompt_version', default='v1', help='which prompt to use for relevance feedback')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate pointwise reranking')
    parser.add_argument('--init_retrieval_method', default='hybrid', help='search method for initial retrieval')
    parser.add_argument('--k_init', type=int, default=20, help='number of documents for the LLM-Rf to judge')
    parser.add_argument('--output_filename', required=True,  help='output filename')
    args = parser.parse_args()
    ###############################
    # Load queries
    qids, queries = load_queries_qids(args.corpus_name)
    ###############################
    # Load searchers
    lucence_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])
    encoder = AutoQueryEncoder(encoder_dir=args.query_encoder_path, pooling='mean')
    if args.corpus_name == 'dl19' or args.corpus_name == 'dl20':
        faiss_searcher = FaissSearcher(THE_DENSE_INDEX[args.corpus_name], encoder)
    else:
        faiss_searcher = FaissSearcher.from_prebuilt_index(THE_DENSE_INDEX[args.corpus_name], encoder)
    ###############################
    # Load prompter
    prompter = Promptor(task=args.corpus_name, relevance_prompt=args.prompt_version)
    ###############################
    # Load instruction LLM 
    if 'mistral' in args.instruction_model_path.lower() or 'mixtral' in args.instruction_model_path.lower():
        print("Using Mistral!")
        instruction_llm = MistralGenerator(model_path=args.instruction_model_path)
    if 'gemma' in args.instruction_model_path.lower():
        print("Using Gemma!")
        instruction_llm = GemmaGenerator(model_path=args.instruction_model_path)
    if 'meta-llama' in args.instruction_model_path.lower():
        print("Using Llama3!")
        instruction_llm = Llama3Generator(model_path=args.instruction_model_path)
    ###############################
    # Create ReDE-RF object
    pointwise_reranker = PointwiseRerank(prompter=prompter,
                                         encoder=encoder, 
                                         sparse_searcher=lucence_searcher, 
                                         faiss_searcher=faiss_searcher, 
                                         instruction_llm=instruction_llm, 
                                         init_retrieval_method=args.init_retrieval_method)
    ###############################
    # Run pointwise reranking
    output_filename = f'{args.output_filename}_pointwise_reranking_{args.init_retrieval_method}_k_init-{args.k_init}_{args.corpus_name}'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = pointwise_reranker.e2e_search(query=query, 
                                                 k_init=args.k_init)
            if hits is not None:
                rank = 0
                for hit in hits:
                    rank += 1
                    try:
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    except:
                        f.write(f'{qid} Q0 {hit["docid"]} {rank} {hit["score"]} rank\n')
    ###############################
    # Eval pointwise reranking
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.20 {THE_TOPICS[args.corpus_name]} {output_filename}"))