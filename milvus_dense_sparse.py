import random
import string
from pymilvus import utility, connections, Collection
from milvus_util import create_connection, create_collection, get_embeddings, load_entries, delete_entries, hybrid_search, rerank_results

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

def main():
    create_connection()

    query = "Who started AI research?"
    docs_embeddings, query_embeddings, dense_dim = get_embeddings(docs, query)
    
    col_name = "hybrid"
    
    if col_name not in utility.list_collections():
        col = create_collection(col_name, dense_dim, docs, docs_embeddings)
    else:
        col = Collection(name=col_name)
        load_entries(col_name, docs, docs_embeddings)
        # delete_entries(col_name, [
        #     "451173335721318074",
        #     "451173335721318034",
        #     "451173335721318024",
        #     "451173335721318054",
        #     "451173335721318064"
        # ])

    k = 30
    top_k = 5
    res = hybrid_search(col, query_embeddings, k)
    rerank_results(query, res, k, top_k)
    
    # utility.drop_collection(col.name)

if __name__ == "__main__":
    main()
