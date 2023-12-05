from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["k pop"]

for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
        print(f"{doc:30} {score:>.3f}")
    print()
    # for (score, doc) in BSBI_instance.retrieve_bm25(query):
    #     print(f"{doc:30} {score:>.3f}")
    # print()
