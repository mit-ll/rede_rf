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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Contriever')
    parser.add_argument('--query_encoder_path', required=True, help='base model path')
    parser.add_argument('--corpus_name', required=True, help='base model path')
    parser.add_argument('--k', type=int, default=20, help='search results for final retrieval')
    parser.add_argument('--do_hybrid',  action='store_true')
    parser.add_argument('--output_filename', required=True)
    args = parser.parse_args()

    qids, queries = load_queries_qids(args.corpus_name)
    lucence_searcher = LuceneSearcher.from_prebuilt_index(THE_SPARSE_INDEX[args.corpus_name])

    encoder = AutoQueryEncoder(encoder_dir=args.query_encoder_path, pooling='mean', l2_norm=False)
    faiss_searcher = FaissSearcher.from_prebuilt_index(THE_DENSE_INDEX[args.corpus_name], encoder)

    if args.do_hybrid:
        hybrid_searcher = HybridSearcher(faiss_searcher, lucence_searcher)
        
    output_filename = f'{args.output_filename}_contriever'
    with open(output_filename, 'w')  as f:
        for idx in tqdm(range(len(queries))):
            query = queries[idx]
            qid = qids[idx]
            if args.do_hybrid:
                hits = hybrid_searcher.search(query, k0=args.k, k=args.k)
            else:
                hits = faiss_searcher.search(query, k=args.k)
            rank = 0
            for hit in hits:
                rank += 1
                f.write(f'{qid} Q0 {hit.docid} {rank} {hit.score} rank\n')

    print(os.system(f"python -m pyserini.eval.trec_eval -c -m ndcg_cut.10 {THE_TOPICS[args.corpus_name]} {output_filename}"))
