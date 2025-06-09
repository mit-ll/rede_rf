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
from itertools import chain

import pandas as pd
from index_paths import THE_TOPICS, THE_SPARSE_INDEX, THE_DENSE_INDEX, load_queries_qids

import nltk
import json 
from numpy import dot
from numpy.linalg import norm
from rf import LLM_RF
from rede_rf import REDE_RF
from prompter import Promptor
from generator import MistralGenerator, GemmaGenerator, Llama3Generator
from transformers.utils import logging
logging.set_verbosity_error() 

class CSQE(REDE_RF):
    def __init__(self, 
                 prompter, 
                 encoder, 
                 sparse_searcher, 
                 faiss_searcher, 
                 instruction_llm, 
                 init_retrieval_method='hybrid'):
        self.prompter = prompter
        self.instruction_llm = instruction_llm
        self.encoder = encoder

        self.faiss_searcher = faiss_searcher
        # sparse searcher needed for loading original document text and for hybrid search, if needed
        self.sparse_searcher = sparse_searcher
        self.init_retrieval_method = init_retrieval_method
        if self.init_retrieval_method == 'hybrid':
            assert(self.faiss_searcher is not None)
            self.hybrid_searcher = HybridSearcher(self.faiss_searcher, self.sparse_searcher )
    
    def split_into_sentences(self, passage):
        return nltk.tokenize.sent_tokenize(passage)

    def judge_documents(self, query, retrieved_sentences):
        relevant_sentences = []
        for sentence in retrieved_sentences:
            prompt, class_names = self.prompter.build_relevance_assessment_prompt(passage=sentence, query=query)
            response = self.instruction_llm.generate_assessments(prompt, class_names)
            prediction, score = response.argmax().item(), response[-1].item()
            if prediction == 1:
                relevant_sentences.append(sentence)      
        return relevant_sentences

    def e2e_search(self, query, qid=None, k_init=20, k=10):
        retrieved_documents = self.initial_retrieval(query, k_init=k_init)
        context_sentences = [self.split_into_sentences(self.instruction_llm.truncate(passage, length=128)) for passage, docid, search_type in retrieved_documents]
        context_sentences = list(chain.from_iterable(context_sentences))
        relevant_sentences = self.judge_documents(query=query, retrieved_sentences=context_sentences)
        prompt = self.prompter.build_passage_generation_prompt_hyde(query)
        hypothesis_documents = self.instruction_llm.generate_synthetic_passages(prompt, num_return_sequences=2)
        query_vector = self.encoder.encode([query])
        relevant_sentences += hypothesis_documents
        # As relevant_sentences needed to be encoded by Contriever (since it is not pre-computed), we set
        # is_hypothetical_docs=True so it gets encoded.
        csqe_vector = self.encode_rede_vector(query_vector, relevant_sentences, is_hypothetical_docs=True)
        hits = self.faiss_searcher.search(csqe_vector, k=k)
        return hits

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CSQE')
    parser.add_argument('--instruction_model_path', help='instruction model path')
    parser.add_argument('--query_encoder_path', default='facebook/contriever', help='query encoder path')
    parser.add_argument('--prompt_version', default='v1', help='which prompt to use for relevance feedback')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate rede-rf')
    parser.add_argument('--init_retrieval_method', default='hybrid', help='search method for initial retrieval')
    parser.add_argument('--k_init', type=int, default=20, help='number of sentences for the LLM-Rf to judge')
    parser.add_argument('--k', type=int, default=20, help='number of documents to return for final retrieval')
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
    # Create CSQE object
    csqe = CSQE(prompter=prompter,
                instruction_llm=instruction_llm, 
                encoder=encoder, 
                sparse_searcher=lucence_searcher, 
                faiss_searcher=faiss_searcher, 
                init_retrieval_method=args.init_retrieval_method)
    ###############################
    # Run CSQE
    output_filename = f'{args.output_filename}_csqe_{args.init_retrieval_method}_k_init-{args.k_init}_{args.corpus_name}'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = csqe.e2e_search(query=query, k_init=args.k_init, k=args.k)
            if hits is not None:
                rank = 0
                for hit in hits:
                    rank += 1
                    try:
                        f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
                    except:
                        f.write(f'{qid} Q0 {hit["docid"]} {rank} {hit["score"]} rank\n')
    ###############################
    # Eval CSQE
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))