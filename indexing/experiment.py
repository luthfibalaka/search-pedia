import os
import math

from bsbi import BSBIIndex
from letor import Letor
from compression import VBEPostings
from tqdm import tqdm
from collections import defaultdict


def rbp(ranking, p=0.8):
    """menghitung search effectiveness metric score dengan
    Rank Biased Precision (RBP)

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score RBP
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking: list[int]):
    """menghitung search effectiveness metric score dengan
    Discounted Cumulative Gain

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score DCG
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        rank_weight = math.log2(i + 1)
        score += ranking[i - 1] / rank_weight
    return score


def prec(ranking, k):
    """menghitung search effectiveness metric score dengan
    Precision at K

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    k: int
      banyak dokumen yang dipertimbangkan atau diperoleh

    Returns
    -------
    Float
      score Prec@K
    """
    score = 0.0
    for i in range(1, len(ranking) + 1):
        score += ranking[i - 1]
    return score / k


def ap(ranking: list[int]):
    """menghitung search effectiveness metric score dengan
    Average Precision

    Parameters
    ----------
    ranking: List[int]
       vektor biner seperti [1, 0, 1, 1, 1, 0]
       gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
       Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
               di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
               di rank-6 tidak relevan

    Returns
    -------
    Float
      score AP
    """
    try:
        # Approximating R
        R = 0
        for rank in ranking:
            R += rank

        # Count the AP
        score = 0.0
        for i in range(1, len(ranking) + 1):
            score += prec(ranking, len(ranking)) * ranking[i - 1]
        return score / R
    except ZeroDivisionError:  # Kasus saat tidak ada hasil yang relevan
        return 0


def eval_retrieval(qrels, query_file="qrels_folder/queries.txt", k=100):
    """
    loop ke semua query, hitung score di setiap query,
    lalu hitung MEAN SCORE-nya. untuk setiap query, kembalikan top-100 documents
    TAMBAHAN: Ada mencoba re-ranking juga menggunakan LETOR
    """
    BSBI_instance = BSBIIndex(
        data_dir="collections", postings_encoding=VBEPostings, output_dir="index"
    )
    letor_instance = Letor("qrels_folder", BSBI_instance)

    with open(query_file) as file:
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []

        rbp_scores_tfidf_updated = []
        dcg_scores_tfidf_updated = []
        ap_scores_tfidf_updated = []

        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []

        rbp_scores_bm25_updated = []
        dcg_scores_bm25_updated = []
        ap_scores_bm25_updated = []

        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])

            """
            a. Evaluasi TF-IDF
            """
            ranking_tfidf = []
            data_for_reranking = []  # Nyimpen informasi untuk re-ranking
            for _, doc in BSBI_instance.retrieve_tfidf(query, k=k):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if did in qrels[qid]:
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
                
                # Simpan informasi untuk re-ranking
                with open(doc, encoding="utf8") as doc_file:
                    content = doc_file.readline()
                    data_for_reranking.append((did, content))
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            b. Evaluasi TF-IDF re-ranked by LETOR
            """
            updated_ranking_tfidf = []
            re_ranked_tfidf_docs_scores = letor_instance.re_ranking(
                query, data_for_reranking
            )
            for did, _ in re_ranked_tfidf_docs_scores:
                if did in qrels[qid]:
                    updated_ranking_tfidf.append(1)
                else:
                    updated_ranking_tfidf.append(0)

            rbp_scores_tfidf_updated.append(rbp(updated_ranking_tfidf))
            dcg_scores_tfidf_updated.append(dcg(updated_ranking_tfidf))
            ap_scores_tfidf_updated.append(ap(updated_ranking_tfidf))

            """
            c. Evaluasi BM25 (k1=1.2, b=0.75)
            """
            ranking_bm25 = []
            data_for_reranking = []  # Nyimpen informasi untuk re-ranking

            for _, doc in BSBI_instance.retrieve_bm25(query, k=k):
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if did in qrels[qid]:
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
                
                # Simpan informasi untuk re-ranking
                with open(doc, encoding="utf8") as doc_file:
                    content = doc_file.readline()
                    data_for_reranking.append((did, content))
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            """
            d. Evaluasi BM25 re-ranked by LETOR
            """
            updated_ranking_bm25 = []
            re_ranked_bm25_docs_scores = letor_instance.re_ranking(
                query, data_for_reranking
            )
            for did, _ in re_ranked_bm25_docs_scores:
                if did in qrels[qid]:
                    updated_ranking_bm25.append(1)
                else:
                    updated_ranking_bm25.append(0)

            rbp_scores_bm25_updated.append(rbp(updated_ranking_bm25))
            dcg_scores_bm25_updated.append(dcg(updated_ranking_bm25))
            ap_scores_bm25_updated.append(ap(updated_ranking_bm25))

    print("Hasil evaluasi TF-IDF terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
    print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
    print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

    print("Hasil evaluasi TF-IDF re-ranked terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_tfidf_updated) / len(rbp_scores_tfidf_updated))
    print("DCG score =", sum(dcg_scores_tfidf_updated) / len(dcg_scores_tfidf_updated))
    print("AP score  =", sum(ap_scores_tfidf_updated) / len(ap_scores_tfidf_updated))

    print("Hasil evaluasi BM25 (k1=1.2, b=0.75) terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
    print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
    print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

    print("Hasil evaluasi BM25 (k1=1.2, b=0.75) re-ranked terhadap 150 queries")
    print("RBP score =", sum(rbp_scores_bm25_updated) / len(rbp_scores_bm25_updated))
    print("DCG score =", sum(dcg_scores_bm25_updated) / len(dcg_scores_bm25_updated))
    print("AP score  =", sum(ap_scores_bm25_updated) / len(ap_scores_bm25_updated))


def load_qrels(qrel_file="qrels_folder/test_qrels.txt"):
    """
    Load qrels dalam format dictionary of dictionary.
    Misal: {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    qrels = defaultdict(lambda: defaultdict(lambda: 0))
    with open(qrel_file) as file:
        for line in file:
            parts = line.strip().split()
            qid = parts[0]
            did = int(parts[1])
            qrels[qid][did] = 1
    return qrels


if __name__ == "__main__":
    qrels = load_qrels()
    eval_retrieval(qrels)
