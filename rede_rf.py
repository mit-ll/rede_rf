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

class REDE_RF(LLM_RF):
    def __init__(self, 
                 prompter, 
                 encoder, 
                 sparse_searcher, 
                 faiss_searcher, 
                 instruction_llm, 
                 init_retrieval_method='hybrid', 
                 default='hyde_prf', 
                 rerank_rede=False):

        super().__init__(prompter, instruction_llm, encoder, faiss_searcher, sparse_searcher, init_retrieval_method)

        if default not in ['None', 'query_encoder','hyde_prf']:
            raise ValueError(f"Error: Invalid choice for default. Valid choices are: {['None',  'query_encoder', 'hyde_prf']}")

        self.default = default
        self.rerank_rede = rerank_rede

    def encode_rede_vector(self, query_vector, relevant_docs, is_hypothetical_docs):
        if is_hypothetical_docs:
            print("Encoding HyDE hypothetical documents!")
            rede_rel_docs_vectors = [self.encoder.encode([pseudo_doc]) for pseudo_doc in relevant_docs]
        else:
            rede_rel_docs_vectors = [self.load_document_vector(doc_id) for doc_id in relevant_docs]

        rede_rel_docs_vectors.append(query_vector)
        rede_rel_docs_vectors = np.array(rede_rel_docs_vectors)
        rede_vector = np.mean(rede_rel_docs_vectors, axis=0)
        rede_vector = rede_vector.reshape((1, len(rede_vector)))
        return rede_vector

    def generate_synthetic_passages(self, query, retrieved_documents):
        context = [self.instruction_llm.truncate(passage, length=128) for passage, docid, search_type in retrieved_documents]
        context_formatted = "\n".join([f"Passage {n}: {doc}" for n, doc in enumerate(context)])
        prompt = self.prompter.build_passage_generation_prompt_hyde_prf(query=query, context=context_formatted)
        synthetic_relevant_docs = self.instruction_llm.generate_synthetic_passages(prompt) 
        return synthetic_relevant_docs

    def e2e_search(self, query, k_init, k_star, k):
        retrieved_documents = self.initial_retrieval(query, k_init)
        relevant_docs, non_relevant_docs, _ = self.judge_documents(query, retrieved_documents)
        synthetic_relevant_docs = []

        if relevant_docs == [] and self.default == 'None':
            print("Not returning results!")
            rede_hits = None
        else:
            query_vector = self.encoder.encode([query])
            if relevant_docs == []:
                print(f"No relevant documents in top {len(retrieved_documents)}") 
                if self.default == 'query_encoder':
                    assert(self.default == 'query_encoder')
                    print("Defaulting to query_encoder!")
                    rede_vector = query_vector.reshape((1, len(query_vector)))
                if self.default == 'hyde_prf':
                    print("Generating a synthetic document!")
                    assert(self.default == 'hyde_prf')
                    synthetic_relevant_docs = self.generate_synthetic_passages(query, retrieved_documents)
                    rede_vector = self.encode_rede_vector(query_vector=query_vector, 
                                                          relevant_docs=synthetic_relevant_docs, 
                                                          is_hypothetical_docs=True)
            else:
                k_star_relevant_docs = relevant_docs[:k_star]
                rede_vector = self.encode_rede_vector(query_vector=query_vector, 
                                                      relevant_docs=k_star_relevant_docs, 
                                                      is_hypothetical_docs=False)

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
    parser = argparse.ArgumentParser(description='Evaluate ReDE-RF')
    parser.add_argument('--instruction_model_path', help='instruction model path')
    parser.add_argument('--query_encoder_path', default='facebook/contriever', help='query encoder path')
    parser.add_argument('--prompt_version', default='v1', help='which prompt to use for relevance feedback')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate rede-rf')
    parser.add_argument('--init_retrieval_method', default='hybrid', help='search method for initial retrieval')
    parser.add_argument('--k_init', type=int, default=20, help='number of documents for the LLM-Rf to judge')
    parser.add_argument('--k', type=int, default=20, help='number of documents to return for final retrieval')
    parser.add_argument('--k_star', type=int, default=None, help='number of documents to return for final retrieval')
    parser.add_argument('--default', default='hyde_prf', choices=['None', 'query_encoder', 'hyde', 'hyde_prf'], help='what to do if rede finds no relevant documents')
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
    rede_rf = REDE_RF(prompter=prompter,
                      encoder=encoder, 
                      sparse_searcher=lucence_searcher, 
                      faiss_searcher=faiss_searcher, 
                      instruction_llm=instruction_llm, 
                      init_retrieval_method=args.init_retrieval_method, 
                      default=args.default,
                      rerank_rede=args.rerank_rede)
    ###############################
    # Define k_star
    if args.k_star is None:
        k_star = args.k_init
    else:
        k_star = args.k_star
    # Run ReDE-RF
    output_filename = f'{args.output_filename}_rede-rf_{args.init_retrieval_method}_k_init-{args.k_init}_default-{args.default}_{args.corpus_name}'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = rede_rf.e2e_search(query=query, 
                                      k_init=args.k_init, 
                                      k_star=k_star,
                                      k=args.k)
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
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.20 {THE_TOPICS[args.corpus_name]} {output_filename}"))