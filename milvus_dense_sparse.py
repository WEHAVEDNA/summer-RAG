import random
import string
from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)


def generate_docs():
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England.",
        "The Turing Test, proposed by Alan Turing in 1950, evaluates a machine's ability to exhibit intelligent behavior.",
        'John McCarthy, a prominent figure in AI, coined the term "artificial intelligence" in 1956.',
        "The Dartmouth Conference, held in 1956, is considered the birthplace of AI as a field of study.",
        "Norbert Wiener's cybernetics described control and stability in electrical networks.",
        'Alan Turing was among the first people to seriously investigate the theoretical possibility of "machine intelligence".',
        'In 1950 Turing published a landmark paper "Computing Machinery and Intelligence", in which he speculated about the possibility of creating machines that think.',
    ]
    docs.extend([' '.join(''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, 8))) for _ in range(10)) for _ in range(1000)])
    
    return docs

def get_embeddings(docs, query):
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction # type: ignore
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = ef.dim["dense"]

    docs_embeddings = ef(docs)
    query_embeddings = ef([query])
    
    return docs_embeddings, query_embeddings, dense_dim

def create_connection():
    connections.connect("default", host="localhost", port="19530")

def prepare_collection(dense_dim, docs, docs_embeddings, col_name):
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=100),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
    ]
    schema = CollectionSchema(fields, "")
    col = Collection(col_name, schema, consistency_level="Strong")

    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    col.create_index("sparse_vector", sparse_index)
    dense_index = {"index_type": "FLAT", "metric_type": "IP"}
    col.create_index("dense_vector", dense_index)
    col.load()

    entities = [docs, docs_embeddings["sparse"], docs_embeddings["dense"]]
    col.insert(entities)
    col.flush()

    return col

def hybrid_search(col, query_embeddings, k=20):
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=k)
    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(query_embeddings["dense"], "dense_vector", dense_search_params, limit=k)
    res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(), limit=k, output_fields=['text'])
    return res[0]

def rerank_results(query, res, k, top_k):
    top_k = min(k, top_k)
    result_texts = [hit.fields["text"] for hit in res]
    from pymilvus.model.reranker import BGERerankFunction # type: ignore
    bge_rf = BGERerankFunction(device='cuda')
    results = bge_rf(query, result_texts, top_k=top_k)
    for hit in results:
        print(f'text: {hit.text} distance {hit.score}')

def main():
    docs = generate_docs()
    query = "Who started AI research?"
    docs_embeddings, query_embeddings, dense_dim = get_embeddings(docs, query)
    
    create_connection()
    col = prepare_collection(dense_dim, docs, docs_embeddings, col_name="hybrid")
    
    k = 30
    top_k = 5
    res = hybrid_search(col, query_embeddings, k)
    rerank_results(query, res, k, top_k)
    
    utility.drop_collection(col.name)

if __name__ == "__main__":
    main()
