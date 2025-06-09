"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

import os
import pickle
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
from prompter import Promptor
from generator import MistralGenerator, GemmaGenerator, Llama3Generator
from transformers.utils import logging

logging.set_verbosity_error() 

class HyDE:
    def __init__(self, prompter, generator, encoder, searcher):
        self.prompter = prompter
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
        
    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector
    
    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        return hits
    
    def e2e_search(self, query, k=10, num_hypothetical_documents=8):
        prompt = self.prompter.build_passage_generation_prompt_hyde(query)
        hypothesis_documents = self.generator.generate_synthetic_passages(prompt, 
                                                       num_return_sequences=num_hypothetical_documents)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.searcher.search(hyde_vector, k=k)
        return hits

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate HYDE')
    parser.add_argument('--instruction_model_path', help='instruction model path')
    parser.add_argument('--query_encoder_path', help='query encoder path')
    parser.add_argument('--corpus_name', required=True, help='corpus to evaluate hyde')
    parser.add_argument('--k', type=int, default=10, help='number of documents to return for final retrieval')
    parser.add_argument('--num_hypothetical_documents', type=int, default=8, help='number of hypothetical documents to generate')
    parser.add_argument('--output_filename', required=True)
    args = parser.parse_args()
    ###############################
    # Load queries
    qids, queries = load_queries_qids(args.corpus_name)
    ###############################
    # Load searchers
    encoder = AutoQueryEncoder(encoder_dir=args.query_encoder_path, pooling='mean')
    if args.corpus_name == 'dl19' or args.corpus_name == 'dl20':
        faiss_searcher = FaissSearcher(THE_DENSE_INDEX[args.corpus_name], encoder)
    else:
        faiss_searcher = FaissSearcher.from_prebuilt_index(THE_DENSE_INDEX[args.corpus_name], encoder)
    ###############################
    # Load prompter
    prompter = Promptor(task=args.corpus_name)
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
    # Create HyDE object
    hyde = HyDE(prompter=prompter,
                generator=instruction_llm,
                encoder=encoder,
                searcher=faiss_searcher,
                )
    ###############################
    # Run HyDE
    output_filename = f'{args.output_filename}_HyDE'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            hits = hyde.e2e_search(query, k=args.k, num_hypothetical_documents=args.num_hypothetical_documents)
            rank = 0
            for hit in hits:
                rank += 1
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')
    ###############################
    # Eval HyDE
    print(f"Results for HyDE for corpus = {args.corpus_name}:")
    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))