import random
import lightgbm
import numpy as np
import logging
import joblib

from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine, minkowski
from bsbi import BSBIIndex

class Letor:
    """
    Class ini mengimplementasikan fungsionalitas LETOR dengan proses:
    1. Persiapan data untuk re-ranking
    2. Pembuatan model LSI/LSA
    3. Melatih LightGBM LGBMRanker
    4. Melakukan prediksi (re-ranking)

    - Untuk training (proses 1-3), panggil train_letor(ranker_trained=False). Model otomatis di-save
    - Jika model sudah di-save, panggil train_letor(ranker_trained=True)
    - Untuk re-ranking (proses 4), panggil re_ranking()
    - Secara khusus, train_letor() digunakan dengan menjalankan file ini
    - sedangkan re_ranking digunakan di experiment.py
    Referensi: https://colab.research.google.com/drive/1r3UzswQgSdScukzSOrbgNoml0TlS04f_?usp=sharing
    """

    def __init__(self, path_to_qrels: str, bsbi_instance: BSBIIndex) -> None:
        """
        Diasumsikan pada folder path_to_qrels, terdapat
        file-file dengan format {train/test/val}_{qrels/queries/docs}.txt
        """
        self.path_to_qrels = path_to_qrels
        self.NUM_LATENT_TOPICS = 200
        self.preprocessor = bsbi_instance.pre_processing_text

        # Setup logger
        logging.basicConfig()
        self.logger = logging.getLogger("letor")
        self.logger.setLevel(logging.INFO)

    def train_letor(self, ranker_trained=False, lsi_trained=False):
        """
        Ini adalah fungsi yang menjalankan proses letor 1-3 secara sekuensial
        """
        self.logger.info("==========Starting letor training!==========")

        if not lsi_trained:
            # 1. Persiapan data training, testing, dan validation
            self.logger.info("Step 1: Preparing training, testing, & validation data")
            self.docs_contents = self.__get_docs_contents(
                f"{self.path_to_qrels}/train_docs.txt"
            )
            self.train_queries_contents = self.__get_queries_contents()

            self.train_dataset = self.__get_dataset(
                f"{self.path_to_qrels}/train_qrels.txt", self.train_queries_contents
            )

            # 2. Membuat model LSI/LSA
            self.logger.info("Step 2: Initializing LSI Model")
            self.lsi_model = self.__build_lsi_model()
        else:
            self.logger.info("Step 1-2: Load trained LSI Model")
            loaded_letor = joblib.load("qrels_folder/letor.pkl")
            self.__dict__.update(loaded_letor.__dict__)

        # 3. Training LightGBM LGBMRanker
        if not ranker_trained:
            self.logger.info("Step 3: Training LightGBM Ranker")

            X, Y = self.__convert_dataset_to_X_Y(
                self.train_dataset, self.train_queries_contents
            )
            self.ranker = lightgbm.LGBMRanker(
                objective="lambdarank",
                boosting_type="gbdt",
                n_estimators=250,
                importance_type="gain",
                metric="ndcg",
                num_leaves=10,
                learning_rate=0.05,
                max_depth=-1,
            )
            self.ranker.fit(X, Y, group=self.train_group_qid_count)

            joblib.dump(self.ranker, "qrels_folder/ranker.pkl")
            self.logger.info("==> Best model tuned and saved")
        else:
            self.logger.info("Step 3: Load trained LightGBM Ranker")
            self.ranker = joblib.load("qrels_folder/ranker.pkl")

        self.logger.info(
            "==========Training finished, saving Letor instance!=========="
        )
        joblib.dump(self, "qrels_folder/letor.pkl")

    def re_ranking(self, query: str, docs: list[tuple[str, str]]):
        """
        Melakukan re-ranking pada suatu query
        """

        # 1. Load Letor
        loaded_letor = joblib.load("qrels_folder/letor.pkl")
        self.__dict__.update(loaded_letor.__dict__)

        # 2. Persiapan data
        X = []
        for _, doc in docs:
            X.append(
                self.__get_features(
                    self.preprocessor(query.strip()),
                    self.preprocessor(doc.strip()),
                    self.train_queries_contents,
                    self.lsi_model,
                )
            )
        X = np.array(X)

        if len(X) == 0:
            return []

        # 3. Prediksi skor
        scores = self.ranker.predict(X)

        # 4. Re-ranking
        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        return sorted(did_scores, key=lambda tup: tup[1], reverse=True)

    def __get_docs_contents(self, docs: str):
        """
        Mengembalikan {doc_id: contents} dari train_docs
        """
        self.logger.info(f"==> Extracting docs contents from train_docs")
        docs_contents: dict[int, list[str]] = {}

        # 1. Extract from train_docs
        with open(docs, encoding="utf8") as docs:
            for example in docs:
                words = self.preprocessor(example.strip())
                docs_contents[int(words[0])] = words[1:]
        return docs_contents

    def __get_queries_contents(self):
        """
        Inisialisasi {query_id: contents} untuk training data
        """
        self.logger.info("==> Extracting training queries")
        train_queries_contents = {}
        with open(
            f"{self.path_to_qrels}/queries.txt", encoding="utf8"
        ) as train_queries:
            for train_example in train_queries:
                words = self.preprocessor(train_example.strip())
                query_id = words[0]
                contents = words[1:]
                train_queries_contents[query_id] = contents
        return train_queries_contents

    def __get_dataset(self, qrels: str, queries_contents: dict):
        """
        Inisialisasi dataset [(query_text, document_text, relevance), ...]
        """
        self.logger.info(f"==> Creating dataset from {qrels}")

        # 1. Grouping berdasarkan query_id
        query_docs_relevance: dict[str, list] = {}
        dataset = []
        with open(qrels, encoding="utf8") as qrels_file:
            for row in qrels_file:
                words = self.preprocessor(row.strip())
                query_id, doc_id, relevance = words[0], int(words[1]), int(words[2])
                if (query_id in queries_contents) and (doc_id in self.docs_contents):
                    if query_id not in query_docs_relevance:
                        query_docs_relevance[query_id] = []
                    query_docs_relevance[query_id].append((doc_id, relevance))

        # 2. Menghasilkan dataset
        NUM_NEGATIVES = 1
        self.train_group_qid_count: list[int] = []
        for query_id in query_docs_relevance:
            docs_rels = query_docs_relevance[query_id]
            self.train_group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                dataset.append(
                    (
                        queries_contents[query_id],
                        self.docs_contents[doc_id],
                        rel,
                    )
                )
            # tambahkan satu negative (random sampling saja dari documents)
            dataset.append(
                (
                    queries_contents[query_id],
                    random.choice(list(self.docs_contents.values())),
                    0,
                )
            )

        return dataset

    def __build_lsi_model(self):
        """
        Membangun model LSI untuk merepresentasikan dokumen pada suatu ruang vektor.
        Queries juga akan dipetakan ke ruang vektor yang sama.
        """
        self.logger.info("==> Training LSI Model")
        self.dictionary = Dictionary()
        bow_corpus = [
            self.dictionary.doc2bow(doc, allow_update=True)
            for doc in self.docs_contents.values()
        ]
        lsi_model = LsiModel(bow_corpus, num_topics=self.NUM_LATENT_TOPICS)
        return lsi_model

    def __convert_dataset_to_X_Y(self, dataset, queries_contents):
        """
        Mengembalikan dataset yang sudah di-convert menjadi X dan Y
        """
        self.logger.info("==> Splitting dataset into X and Y")
        X = []
        Y = []
        for query, doc, rel in dataset:
            X.append(self.__get_features(query, doc, queries_contents, self.lsi_model))
            Y.append(rel)
        return np.array(X), np.array(Y)

    def __get_features(self, query, doc, queries_contents, lsi_model):
        """
        Helper function untuk mendapatkan fitur dari suatu instance query & doc
        """
        v_q = self.__vector_rep(queries_contents, lsi_model)
        v_d = self.__vector_rep(doc, lsi_model)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        minkowski_dist = minkowski(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [cosine_dist] + [jaccard] + [minkowski_dist]

    def __vector_rep(self, text: list[str], lsi_model: LsiModel):
        """
        Helper function untuk mendapatkan representasi vektor dari suatu list of words
        """
        rep = [
            topic_value for (_, topic_value) in lsi_model[self.dictionary.doc2bow(text)]
        ]
        return (
            rep
            if len(rep) == self.NUM_LATENT_TOPICS
            else [0.0] * self.NUM_LATENT_TOPICS
        )


if __name__ == "__main__":
    """
    Ini untuk mendapatkan semua data
    """
    letor = Letor("qrels_folder", BSBIIndex("", "", ""))
    letor.train_letor(False, False)
