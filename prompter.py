"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

##############################################################################
WEB_SEARCH = """Please write a passage to answer the question.
Question: {}
Passage:"""

WEB_SEARCH_PRF = """Please write a passage to answer the question based on the context:
Context: 
{}
Question: {}
Passage:"""
##############################################################################
SCIFACT = """Please write a scientific paper passage to support/refute the claim.
Claim: {}
Passage:"""

SCIFACT_PRF = """Please write a scientific paper passage to support/refute the claim based on the context:
Context: 
{}
Claim: {}
Passage:"""
##############################################################################
TREC_COVID = """Please write a scientific paper passage to answer the question.
Question: {}
Passage:"""

TREC_COVID_PRF = """Please write a scientific paper passage to answer the question based on the context:
Context:
{}
Question: {}
Passage:"""
##############################################################################
FIQA = """Please write a financial article passage to answer the question.
Question: {}
Passage:"""

FIQA_PRF = """Please write a financial article passage to answer the question based on the context:
Context: 
{}
Question: {}
Passage:"""
##############################################################################
DBPEDIA_ENTITY = """Please write a passage to answer the question.
Question: {}
Passage:"""

DBPEDIA_ENTITY_PRF = """Please write a passage to answer the question based on the context:
Context: 
{}
Question: {}
Passage:"""
##############################################################################
TREC_NEWS = """Please write a news passage about the topic.
Topic: {}
Passage:"""

TREC_NEWS_PRF = """Please write a news passage about the topic based on the context:
Context: 
{}
Topic: {}
Passage:"""
##############################################################################
RELEVANCE_ASSESSOR_V1 = """You are an expert judge of content. Using your internal knowledge and simple commonsense reasoning, try to verify if the passage is relevant to the query. 
Here, "0" represents that the passage has nothing to do with the query, "1" represents that the passage is dedicated to the query and contains the exact answer.
Instructions: Think about the given query and then provide your answer in terms of 0 or 1 categories. Only provide the relevance category on the last line. Do not provide any 
further details on the last line. 
Passage: {}
Query: {}
Relevance category:"""

###############################

RELEVANCE_ASSESSOR_V2 = """
Passage: {}
Query: {}
Does the passage answer the query? Answer 'Yes' or 'No'.
"""

###############################

RELEVANCE_ASSESSOR_V3 = """For the following query and document, judge whether they are relevant. Output “Yes” or “No”.
Query: {}
Document: {}
Output:"""

###############################

RELEVANCE_ASSESSOR_V4 = """You are a search quality rater evaluating the relevance of web pages. Given a query and a web page, you must
provide a score on an integer scale of 0 to 1 with the following meanings:
1 = highly relevant, very helpful for this query
0 = not relevant, should never be shown for this query
Assume that you are writing a report on the subject of the topic. If the web page is primarily about the topic, or contains vital information about the topic, mark it 1. Otherwise, mark it 0.
Passage: {}
Query: {}
Score:"""

###############################

RELEVANCE_ASSESSOR_V5 = """For the following query and document, judge whether they are relevant. Output “Yes” if the passage is dedicated to the query and contains the exact answer 
and output "No" if the passage has nothing to do with the query.
Query: {}
Document: {}
Output:"""


##############################################################################
class Promptor:
    def __init__(self, task, relevance_prompt='v1') :
        self.task = task
        self.relevance_prompt = relevance_prompt

    def build_passage_generation_prompt_hyde(self, query):
        if self.task == 'dl19' or self.task == 'dl20':
            return WEB_SEARCH.format(query)
        elif self.task == 'scifact':
            return SCIFACT.format(query)
        elif self.task == 'covid' or self.task == 'nfcorpus':
            return TREC_COVID.format(query)
        elif self.task == 'fiqa':
            return FIQA.format(query)
        elif self.task == 'dbpedia':
            return DBPEDIA_ENTITY.format(query)
        elif self.task == 'news' or self.task == 'robust04':
            return TREC_NEWS.format(query)
        else:
            raise ValueError('Task not supported')

    def build_passage_generation_prompt_hyde_prf(self, query, context):
        if self.task == 'dl19' or self.task == 'dl20':
            return WEB_SEARCH_PRF.format(context, query)
        elif self.task == 'scifact':
            return SCIFACT_PRF.format(context, query)
        elif self.task == 'covid' or self.task == 'nfcorpus':
            return TREC_COVID_PRF.format(context, query)
        elif self.task == 'fiqa':
            return FIQA_PRF.format(context, query)
        elif self.task == 'dbpedia':
            return DBPEDIA_ENTITY_PRF.format(context, query)
        elif self.task == 'news' or self.task == 'robust04':
            return TREC_NEWS_PRF.format(context, query)
        else:
            raise ValueError('Task not supported')

    def build_relevance_assessment_prompt(self, passage, query):
        if self.relevance_prompt == 'v1':
            return RELEVANCE_ASSESSOR_V1.format(passage, query), ["0", "1"]
        if self.relevance_prompt == 'v2':
            return RELEVANCE_ASSESSOR_V2.format(passage, query), ["No", "Yes"]
        if self.relevance_prompt == 'v3':
            return RELEVANCE_ASSESSOR_V3.format(query, passage), ["No", "Yes"]
        if self.relevance_prompt == 'v4':
            return RELEVANCE_ASSESSOR_V4.format(passage, query), ["0", "1"]
        if self.relevance_prompt == 'v5':
            return RELEVANCE_ASSESSOR_V5.format(query, passage), ["No", "Yes"]
