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
from rede_rf import REDE_RF
from prompter import Promptor
from generator import MistralGenerator, GemmaGenerator, Llama3Generator
from transformers.utils import logging

logging.set_verbosity_error() 

class REDE_RF_HyDE_Avg(REDE_RF):
    def encode_rede_vector(self, query_vector, relevant_docs, hypothetical_docs):
        hypothetical_doc_vectors = [self.encoder.encode([hypothetical_doc]) for hypothetical_doc in hypothetical_docs]
        real_docs_vectors = [self.load_document_vector(doc_id) for doc_id in relevant_docs]

        rede_pp_vector = hypothetical_doc_vectors + real_docs_vectors
        rede_pp_vector.append(query_vector)
        rede_pp_vector = np.array(rede_pp_vector)
        rede_vector = np.mean(rede_pp_vector, axis=0)
        rede_vector = rede_vector.reshape((1, len(rede_vector)))
        return rede_vector

    def e2e_search(self, query, k_init=20, k=20):
        retrieved_documents = self.initial_retrieval(query, k_init)
        relevant_docs, non_relevant_docs, _ = self.judge_documents(query, retrieved_documents)

        prompt = self.prompter.build_passage_generation_prompt_hyde(query)
        hypothesis_documents = self.instruction_llm.generate_synthetic_passages(prompt, num_return_sequences=2)
        query_vector = self.encoder.encode([query])
        rede_vector = self.encode_rede_vector(query_vector=query_vector, 
                                              relevant_docs=relevant_docs,
                                              hypothetical_docs=hypothesis_documents)
        rede_hits = self.faiss_searcher.search(rede_vector, k=k)
        # Optionally re-rank rede-rf retrieved documents
        if self.rerank_rede:
            print("Reranking ReDE-RF!")
            rede_retrieved_documents = [
                [self.load_document_text(hit.docid), hit.docid, "rede"] for hit in rede_hits
            ]
            _, _, llm_reranking_results_rede_rf = self.judge_documents(query, rede_retrieved_documents)
            rede_hits = llm_reranking_results_rede_rf
        return rede_hits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate ReDE-RF (Avg. HyDE and Relevant Documents)')
    parser.add_argument('--instruction_model_path', help='instruction model path')
    parser.add_argument('--query_encoder_path', default='facebook/contriever', help='query encoder path')
    parser.add_argument('--prompt_version', default='v1', help='which prompt to use for relevance feedback')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate rede-rf')
    parser.add_argument('--init_retrieval_method', default='hybrid', help='search method for initial retrieval')
    parser.add_argument('--k_init', type=int, default=20, help='number of documents for the LLM-Rf to judge')
    parser.add_argument('--k', type=int, default=20, help='number of documents to return for final retrieval')
    parser.add_argument('--rerank_rede', action='store_true')
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
    rede_rf = REDE_RF_HyDE_Avg(prompter=prompter,
                               encoder=encoder, 
                               sparse_searcher=lucence_searcher, 
                               faiss_searcher=faiss_searcher, 
                               instruction_llm=instruction_llm, 
                               init_retrieval_method=args.init_retrieval_method, 
                               default='None',
                               rerank_rede=args.rerank_rede)
    ###############################
    # Run ReDE-RF w/ HyDE Avg!
    output_filename = f'{args.output_filename}_rede-rf-hyde-avg_{args.init_retrieval_method}_k_init-{args.k_init}_{args.corpus_name}'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = rede_rf.e2e_search(query=query, k_init=args.k_init, k=args.k)
            if hits is not None:
                rank = 0
                for hit in hits:
                    rank += 1
                    try:
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    except:
                        f.write(f'{qid} Q0 {hit["docid"]} {rank} {hit["score"]} rank\n')
    ###############################
    # Eval ReDE-RF
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))