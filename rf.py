"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

import json 
from types import SimpleNamespace
from pyserini.search.hybrid import HybridSearcher

class LLM_RF:        
    def __init__(self, prompter, instruction_llm, encoder, faiss_searcher, sparse_searcher, init_retrieval_method):
        self.prompter = prompter
        self.instruction_llm = instruction_llm

        self.encoder = encoder
        self.faiss_searcher = faiss_searcher
        # sparse searcher needed for loading original document text and for hybrid search, if needed
        self.sparse_searcher = sparse_searcher
        self.init_retrieval_method = init_retrieval_method

        if self.init_retrieval_method == 'hybrid':
            assert(self.faiss_searcher is not None)
            self.hybrid_searcher = HybridSearcher(self.faiss_searcher, self.sparse_searcher)

        self.docid_dict = {self.faiss_searcher.docids[i]: i for i in range(len(self.faiss_searcher.docids))}
        
    def load_document_text(self, doc_id):
        content = json.loads(self.sparse_searcher.doc(doc_id).raw())
        if self.prompter.task == 'dl19' or self.prompter.task == 'dl20':
            text = content['contents']
        else:
            text = content['text']
            if 'title' in content:
                text = f'{content["title"]} {text}'
        assert (text != '')
        return text 

    def load_document_vector(self, doc_id):
        faiss_doc_index = self.docid_dict[doc_id]
        return self.faiss_searcher.index.reconstruct(faiss_doc_index)

    def judge_documents(self, query, retrieved_documents):
        relevant_documents = []
        non_relevant_documents = [] 
        all_llm_scores = [] # We also store the LLM scores
        for doc_id_text in retrieved_documents:
            passage, docid, search_type = doc_id_text
            passage = self.instruction_llm.truncate(passage, length=128) 
            prompt, class_names = self.prompter.build_relevance_assessment_prompt(passage=passage, query=query)
            response = self.instruction_llm.generate_assessments(prompt, class_names)
            prediction, score = response.argmax().item(), response[-1].item()
            if prediction == 1:
                relevant_documents += [docid] 
            else:
                non_relevant_documents += [docid] 
            all_llm_scores.append({'docid': docid, 'score': score})
        
        llm_reranking_results = sorted(all_llm_scores, key=lambda x: x['score'], reverse=True)
        return relevant_documents, non_relevant_documents, llm_reranking_results

    def initial_retrieval(self, query, k_init):
        if self.init_retrieval_method == 'hybrid':
            hybrid_hits = self.hybrid_searcher.search(query, k0=k_init, k=k_init)
            retrieved_documents = [[self.load_document_text(hybrid_hits[i].docid), hybrid_hits[i].docid, "hybrid"] for i in range(0, len(hybrid_hits))]

        if self.init_retrieval_method == 'dense':
            query_vector = self.encoder.encode([query])
            query_vector = query_vector.reshape((1, len(query_vector)))
            dense_hits = self.faiss_searcher.search(query_vector, k_init)
            retrieved_documents = [[self.load_document_text(dense_hits[i].docid), dense_hits[i].docid, "dense"] for i in range(0, len(dense_hits))]

        if self.init_retrieval_method == 'sparse':
            sparse_hits = self.sparse_searcher.search(query, k_init)
            retrieved_documents = [[self.load_document_text(sparse_hits[i].docid), sparse_hits[i].docid, "sparse"] for i in range(0, len(sparse_hits))]
            
        return retrieved_documents