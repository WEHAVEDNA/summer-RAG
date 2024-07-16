from pymilvus import (
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, AnnSearchRequest, RRFRanker, connections,
)
import numpy as np

def create_connection():
    connections.connect("default", host="localhost", port="19530")

def get_embeddings(docs, query):
    from pymilvus.model.hybrid import BGEM3EmbeddingFunction # type: ignore
    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cuda")
    dense_dim = ef.dim["dense"]

    docs_embeddings = ef(docs)
    query_embeddings = ef([query])
    
    return docs_embeddings, query_embeddings, dense_dim

def create_collection(col_name, dense_dim, docs, docs_embeddings):
    print(f"Creating collection: {col_name}")

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

    print("Collection updated.")

    return col

def load_entries(col_name, docs, docs_embeddings):
    print(f"Loading entries into collection: {col_name}")

    collection = Collection(name=col_name)
    entities = [docs, docs_embeddings["sparse"], docs_embeddings["dense"]]
    collection.insert(entities)
    collection.flush()

    print("Collection updated.")

def delete_entries(col_name, primary_keys):
    print(f"Deleting entries from collection: {col_name}")

    collection = Collection(name=col_name)
    expr = f"pk in {str(primary_keys)}"
    collection.delete(expr)
    collection.flush()

    print("Collection updated.")

def hybrid_search(col, query_embeddings, k=20):
    sparse_search_params = {"metric_type": "IP"}
    sparse_req = AnnSearchRequest(query_embeddings["sparse"], "sparse_vector", sparse_search_params, limit=k)
    dense_search_params = {"metric_type": "IP"}
    dense_req = AnnSearchRequest(query_embeddings["dense"], "dense_vector", dense_search_params, limit=k)
    res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(), limit=k, output_fields=['pk', 'text'])
    return res[0]

def rerank_results(query, res, k, top_k):
    top_k = min(k, top_k)
    result_texts = [hit.fields["text"] for hit in res]
    result_pks = [hit.fields["pk"] for hit in res]
    from pymilvus.model.reranker import BGERerankFunction # type: ignore
    bge_rf = BGERerankFunction(device='cuda')
    results = bge_rf(query, result_texts, top_k=top_k)
    for hit, pk in zip(results, result_pks):
        print(f'pk: {pk} text: {hit.text} distance {hit.score}')

# create_connection()

# docs = ["Document 1 text", "Document 2 text", "Document 3 text"]
# query = "Sample query text"
# docs_embeddings, query_embeddings, dense_dim = get_embeddings(docs, query)

# col_name = "example_collection"
# if col_name not in utility.list_collections():
#     create_collection(col_name, dense_dim, docs, docs_embeddings)

# load_entries(col_name, docs, docs_embeddings)
# delete_entries(col_name, ["0", "1"])  # Uncomment to delete entries

# col = Collection(name=col_name)
# search_results = hybrid_search(col, query_embeddings)
# rerank_results(query, search_results, k=20, top_k=5)