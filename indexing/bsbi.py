import os
import pickle
import contextlib
import heapq
import math
import re
import sys

# Uncomment if you want to run the Django web app
from .indexer import InvertedIndexReader, InvertedIndexWriter
from .util import IdMap, merge_and_sort_posts_and_tfs
from . import util
from .compression import VBEPostings
sys.modules["util"] = util

# Uncomment if you want to train ranking model
# from indexer import InvertedIndexReader, InvertedIndexWriter
# from util import IdMap, merge_and_sort_posts_and_tfs
# from compression import VBEPostings

from tqdm import tqdm
from collections import defaultdict
from mpstemmer import MPStemmer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(
        self, data_dir, output_dir, postings_encoding, index_name="main_index"
    ):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        self.stemmer = MPStemmer()
        self.remover = StopWordRemoverFactory().create_stop_word_remover()

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(os.path.join(os.path.dirname(__file__), "index"), "terms.dict"), "wb") as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(os.path.join(os.path.dirname(__file__), self.output_dir), "docs.dict"), "wb") as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(os.path.join(os.path.dirname(__file__), self.output_dir), "terms.dict"), "rb") as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(os.path.join(os.path.dirname(__file__), self.output_dir), "docs.dict"), "rb") as f:
            self.doc_id_map = pickle.load(f)

    def pre_processing_text(self, content: str):
        """
        Melakukan preprocessing pada text, yakni tokenizing, stemming dan removing stopwords
        """
        # https://github.com/ariaghora/mpstemmer/tree/master/mpstemmer

        preprocessed_content: list[str] = []

        # 1. Tokenize the content (using regex)
        tokenized: list[str] = re.findall(r"\w+", content.lower())

        # 2. Stem and remove stopwords from the tokens
        for token in tokenized:
            stemmed = self.stemmer.stem(token)
            removed = self.remover.remove(stemmed)
            if len(removed) > 0:
                preprocessed_content.append(removed)
        return preprocessed_content

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        td_pairs: list[tuple[int, int]] = []
        filenames_in_block = os.listdir(f"collections/{block_path}")
        for filename in filenames_in_block:
            with open(f"collections/{block_path}/{filename}", encoding="utf8") as document:
                text = document.read()
            tokens = self.pre_processing_text(text)
            doc_id = self.doc_id_map[f"collections/{block_path}/{filename}"]
            for token in tokens:
                token_id = self.term_id_map[token]
                td_pairs.append((token_id, doc_id))
        return td_pairs

    def write_to_index(
        self, td_pairs: list[tuple[int, int]], index: InvertedIndexWriter
    ):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # 1. Ekstraksi informasi dari td_pairs, termasuk komputasi TF
        terms_tf_dict: dict[int, defaultdict[int, int]] = {}  # {termID: {docID: TF}}
        for term_id, doc_id in td_pairs:
            if term_id not in terms_tf_dict:
                terms_tf_dict[term_id] = {}
            if doc_id in terms_tf_dict[term_id]:
                terms_tf_dict[term_id][doc_id] += 1
            else:
                terms_tf_dict[term_id][doc_id] = 1

        # 2. Append informasi ke intermediate index
        for term_id in sorted(terms_tf_dict.keys(), key=lambda x: self.term_id_map[x]): 
            postings_list = sorted(list(terms_tf_dict[term_id].keys()))
            tfs_list = [terms_tf_dict[term_id][doc_id] for doc_id in postings_list]
            index.append(term_id, postings_list, tfs_list)

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: self.term_id_map[x[0]])
        # merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(
                    list(zip(postings, tf_list)), list(zip(postings_, tf_list_))
                )
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

        # 3. Hitung rata-rata panjang dokumen di collection untuk efisiensi query
        avg_doc_length = 0
        for doc_id in merged_index.doc_length.keys():
            avg_doc_length += merged_index.doc_length[doc_id]
        avg_doc_length / len(merged_index.doc_length.keys())
        merged_index.avg_doc_length = avg_doc_length

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        self.load()  # Load dulu mapping terms dan docs
        processed_query_tokens = self.pre_processing_text(query)
        scores: dict[int, int] = defaultdict(int)

        # 1. Akumulasi scoring untuk dokumen-dokumen
        with InvertedIndexReader(
            self.index_name, self.postings_encoding, self.output_dir
        ) as index:
            N = len(index.doc_length)
            for query_term in processed_query_tokens:
                term_id = self.term_id_map[query_term]
                if term_id in index.terms:
                    idf = math.log10(N / index.postings_dict[term_id][1])
                    postings_list, tfs_list = index.get_postings_list(term_id)
                    for i in range(len(postings_list)):
                        normalized_tf = 0
                        if tfs_list[i] > 0:
                            normalized_tf += (1 + (math.log10(tfs_list[i])))
                        scores[postings_list[i]] += (normalized_tf * idf)

        # 2. Mencari dan mengembalikan Top-k document dengan skor tertinggi
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_docs)):
            sorted_docs[i] = (sorted_docs[i][1], self.doc_id_map[sorted_docs[i][0]])
        return sorted_docs[:k]

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        self.load()  # Load dulu mapping terms dan docs
        processed_query_tokens = self.pre_processing_text(query)
        scores: dict[int, int] = defaultdict(int)

        # 1. Akumulasi scoring untuk dokumen-dokumen
        with InvertedIndexReader(
            self.index_name, self.postings_encoding, self.output_dir
        ) as index:
            N = len(index.doc_length)

            for query_term in processed_query_tokens:
                term_id = self.term_id_map[query_term]
                if term_id in index.terms:
                    idf = math.log10(N / index.postings_dict[term_id][1])
                    postings_list, tfs_list = index.get_postings_list(term_id)
                    for i in range(len(postings_list)):
                        numerator = (k1 + 1) * tfs_list[i]
                        denumerator = (
                            k1
                            * (
                                (1 - b)
                                + b
                                * (
                                    index.doc_length[postings_list[i]]
                                    / index.avg_doc_length
                                )
                            )
                            + tfs_list[i]
                        )
                        scores[postings_list[i]] += idf * (numerator / denumerator)

        # 2. Mencari dan mengembalikan Top-k document dengan skor tertinggi
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i in range(len(sorted_docs)):
            sorted_docs[i] = (sorted_docs[i][1], self.doc_id_map[sorted_docs[i][0]])
        return sorted_docs[:k]

    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = "intermediate_index_" + block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(
                index_id, self.postings_encoding, directory=self.output_dir
            ) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None
        self.save()

        with InvertedIndexWriter(
            self.index_name, self.postings_encoding, directory=self.output_dir
        ) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [
                    stack.enter_context(
                        InvertedIndexReader(
                            index_id, self.postings_encoding, directory=self.output_dir
                        )
                    )
                    for index_id in self.intermediate_indices
                ]
                self.merge_index(indices, merged_index)


if __name__ == "__main__":
    BSBI_instance = BSBIIndex(
        data_dir="collections", postings_encoding=VBEPostings, output_dir="index"
    )
    BSBI_instance.do_indexing()  # memulai indexing!
