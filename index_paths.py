"""
Research was sponsored by the Department of the Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. 
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, 
of the Department of the Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any 
copyright notation herein.
"""

from pyserini.search import get_topics, get_qrels

THE_SPARSE_INDEX = {
    'dl19': 'msmarco-v1-passage',
    'dl20': 'msmarco-v1-passage',
    'covid': 'beir-v1.0.0-trec-covid.flat',
    'news': 'beir-v1.0.0-trec-news.flat',
    'scifact': 'beir-v1.0.0-scifact.flat',
    'fiqa': 'beir-v1.0.0-fiqa.flat', 
    'nfcorpus': 'beir-v1.0.0-nfcorpus.flat',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.flat',
    'robust04': 'beir-v1.0.0-robust04.flat'
}

THE_DENSE_INDEX = {
    'dl19': '<add/path/to/contriever_msmarco_index>',
    'dl20': '<add/path/to/contriever_msmarco_index>',
    'covid':  'beir-v1.0.0-trec-covid.contriever',
    'news': 'beir-v1.0.0-trec-news.contriever',
    'scifact': 'beir-v1.0.0-scifact.contriever',
    'fiqa': 'beir-v1.0.0-fiqa.contriever',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity.contriever',
    'nfcorpus': 'beir-v1.0.0-nfcorpus.contriever',
    'robust04': 'beir-v1.0.0-robust04.contriever',
}

THE_TOPICS = {
    'dl19': 'dl19-passage',
    'dl20': 'dl20-passage',
    'covid': 'beir-v1.0.0-trec-covid-test',
    'news': 'beir-v1.0.0-trec-news-test',
    'scifact': 'beir-v1.0.0-scifact-test',
    'fiqa': 'beir-v1.0.0-fiqa-test',
    'dbpedia': 'beir-v1.0.0-dbpedia-entity-test',
    'nfcorpus': 'beir-v1.0.0-nfcorpus-test',
    'robust04': 'beir-v1.0.0-robust04-test',
}

def load_queries_qids(corpus_name):
    topics = get_topics(THE_TOPICS[corpus_name] if corpus_name != 'dl20' else 'dl20')
    qrels = get_qrels(THE_TOPICS[corpus_name])
    test_only_qids_queries = set(qrels.keys())
    topics_qids = [(key, topics[key]['title'])  for key in topics if key in test_only_qids_queries]
    qids = [i[0] for i in topics_qids]
    queries = [i[1] for i in topics_qids]
    return qids, queries