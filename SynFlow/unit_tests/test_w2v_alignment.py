"""Tests for frequency-based Word2Vec alignment anchors."""

from pathlib import Path
import tempfile
import unittest

from gensim.models import KeyedVectors, Word2Vec
import numpy as np

from SynFlow.Embedding.w2v_training import (
    _dependency_region_rotations,
    _load_vocab_frequencies,
    _orthogonal_procrustes_rotation_for_vectors,
    _select_dependency_anchors,
    _select_dependency_anchors_by_relation,
    align_depw2v_raw_vec_folder,
    align_w2v_folder,
    align_w2v_raw_vec_folder,
)


class W2VAlignmentAnchorTests(unittest.TestCase):
    """Check anchor selection and the saved-model alignment workflow."""

    def test_top_k_uses_the_lower_frequency_in_each_pair(self) -> None:
        base = KeyedVectors(vector_size=2)
        other = KeyedVectors(vector_size=2)
        words = ["a", "b", "c"]
        base.add_vectors(words, np.array([[1, 0], [0, 1], [-1, 0]], dtype=np.float32))
        other.add_vectors(words, np.array([[0, 1], [-1, 0], [-1, 0]], dtype=np.float32))
        for word, base_count, other_count in (
            ("a", 100, 90),
            ("b", 80, 70),
            ("c", 1000, 1),
        ):
            base.set_vecattr(word, "count", base_count)
            other.set_vecattr(word, "count", other_count)

        rotation, anchor_count = _orthogonal_procrustes_rotation_for_vectors(
            base_vectors=base,
            other_vectors=other,
            min_anchor_count=2,
            top_k_anchor=2,
        )

        self.assertEqual(anchor_count, 2)
        np.testing.assert_allclose(other.vectors[:2] @ rotation, base.vectors[:2], atol=1e-6)

        _, all_anchor_count = _orthogonal_procrustes_rotation_for_vectors(
            base_vectors=base,
            other_vectors=other,
            min_anchor_count=2,
        )
        self.assertEqual(all_anchor_count, 3)

        _, minimum_anchor_count = _orthogonal_procrustes_rotation_for_vectors(
            base_vectors=base,
            other_vectors=other,
            min_anchor_count=2,
            top_k_anchor=1,
        )
        self.assertEqual(minimum_anchor_count, 2)

        with self.assertWarnsRegex(UserWarning, "only 3.*minimum.*5"):
            _, capped_anchor_count = _orthogonal_procrustes_rotation_for_vectors(
                base_vectors=base,
                other_vectors=other,
                min_anchor_count=5,
                top_k_anchor=1,
            )
        self.assertEqual(capped_anchor_count, 3)

    def test_folder_alignment_keeps_counts_and_reports_selected_anchors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output_root = root / "output"
            sentences_by_period = {
                "t": [["a", "a", "b", "c"]],
                "t1": [["b", "b", "a", "c"]],
                "t2": [["c", "c", "a", "b"]],
            }
            for period, sentences in sentences_by_period.items():
                period_dir = input_root / period
                period_dir.mkdir(parents=True)
                model = Word2Vec(
                    sentences=sentences,
                    vector_size=2,
                    min_count=1,
                    workers=1,
                    epochs=1,
                )
                model.save(str(period_dir / f"{period}.model"))

            results = align_w2v_folder(
                input_root,
                output_root,
                periods=["t", "t1", "t2"],
                save_formats=("model",),
                top_k_anchor=1,
                min_anchor_count=2,
            )

            self.assertEqual([result.anchor_count for result in results], [0, 2, 2])
            self.assertEqual(results[2].aligned_to_period, "t1")
            self.assertEqual(results[1].vocabulary_size, 3)
            aligned = Word2Vec.load(str(results[1].output_path))
            self.assertEqual(aligned.wv.get_vecattr("b", "count"), 2)

            resumed = align_w2v_folder(
                input_root,
                output_root,
                periods=["t", "t1", "t2"],
                save_formats=("model",),
                top_k_anchor=1,
                min_anchor_count=2,
            )
            self.assertEqual([result.anchor_count for result in resumed], [0, 2, 2])

    def test_top_k_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "top_k_anchor must be at least 1"):
            align_w2v_folder("unused", "unused", periods=["t"], top_k_anchor=0)


class RawW2VAlignmentAnchorTests(unittest.TestCase):
    """Check raw-vector alignment with separate period vocabulary counts."""

    def test_raw_folder_uses_frequency_files_across_three_periods(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output_root = root / "aligned"
            period_data = {
                "t": ([[1, 0], [0, 1], [-1, 0]], [100, 80, 1000]),
                "t1": ([[0, 1], [-1, 0], [-1, 0]], [90, 70, 1]),
                "t2": ([[-1, 0], [0, -1], [-1, 0]], [85, 65, 5000]),
            }
            for period, (vectors, counts) in period_data.items():
                period_dir = input_root / period
                frequency_dir = period_dir / "data_prep"
                frequency_dir.mkdir(parents=True)
                keyed_vectors = KeyedVectors(vector_size=2)
                keyed_vectors.add_vectors(
                    ["a", "b", "c"], np.array(vectors, dtype=np.float32)
                )
                keyed_vectors.save_word2vec_format(str(period_dir / f"{period}.txt"))
                (frequency_dir / "vocab.txt").write_text(
                    "".join(
                        f"{word}\t{count}\n"
                        for word, count in zip(("a", "b", "c"), counts)
                    ),
                    encoding="utf-8",
                )

            results = align_w2v_raw_vec_folder(
                input_root,
                output_root,
                periods=["t", "t1", "t2"],
                vocab_freq_filename="data_prep/vocab.txt",
                top_k_anchor=1,
                min_anchor_count=2,
                save_formats=("keyed_vectors",),
            )

            self.assertEqual([result.anchor_count for result in results], [0, 2, 2])
            self.assertEqual(results[2].aligned_to_period, "t1")
            self.assertEqual(results[2].vocabulary_size, 3)
            self.assertFalse((output_root / "t2" / "t2.model").exists())
            base = KeyedVectors.load(str(results[0].output_path))
            last = KeyedVectors.load(str(results[2].output_path))
            np.testing.assert_allclose(last.vectors[:2], base.vectors[:2], atol=1e-6)

            resumed = align_w2v_raw_vec_folder(
                input_root,
                output_root,
                periods=["t", "t1", "t2"],
                vocab_freq_filename="data_prep/vocab.txt",
                top_k_anchor=1,
                min_anchor_count=2,
                save_formats=("keyed_vectors",),
            )
            self.assertEqual([result.anchor_count for result in resumed], [0, 2, 2])

    def test_top_k_requires_complete_frequency_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "vocab_freq_filename is required"):
            align_w2v_raw_vec_folder("unused", "unused", periods=["t"], top_k_anchor=2)

        base = KeyedVectors(vector_size=2)
        other = KeyedVectors(vector_size=2)
        for vectors in (base, other):
            vectors.add_vectors(["a", "b"], np.eye(2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Missing vocabulary frequency"):
            _orthogonal_procrustes_rotation_for_vectors(
                base_vectors=base,
                other_vectors=other,
                min_anchor_count=2,
                top_k_anchor=2,
                base_frequencies={"a": 10},
                other_frequencies={"a": 10, "b": 10},
            )

    def test_vocab_frequency_reader_accepts_tsv_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tsv_path = root / "vocab.txt"
            csv_path = root / "vocab.csv"
            tsv_path.write_text('a\t10\n"/chi_punct\t5\n', encoding="utf-8")
            csv_path.write_text("vocab,frequency\na,10\nb,5\n", encoding="utf-8")
            self.assertEqual(
                _load_vocab_frequencies(tsv_path), {"a": 10, '"/chi_punct': 5}
            )
            self.assertEqual(_load_vocab_frequencies(csv_path), {"a": 10, "b": 5})


class DependencyW2VAlignmentAnchorTests(unittest.TestCase):
    """Check relation quotas, ranking, and undersampling behavior."""

    def test_relation_quota_is_not_redistributed(self) -> None:
        relation_one = ["a/chi_nsubj", "b/chi_nsubj", "c/chi_nsubj"]
        relation_two = ["d/chi_obj", "e/chi_obj", "f/chi_obj", "g/chi_obj"]
        words = relation_one + relation_two
        base = KeyedVectors(vector_size=2)
        other = KeyedVectors(vector_size=2)
        vectors = np.arange(14, dtype=np.float32).reshape(7, 2)
        base.add_vectors(words, vectors)
        other.add_vectors(words, vectors)
        base_frequencies = {
            "a/chi_nsubj": 500,
            "b/chi_nsubj": 200,
            "c/chi_nsubj": 100,
            "base_only/chi_nsubj": 400,
            "d/chi_obj": 100,
            "e/chi_obj": 70,
            "f/chi_obj": 20,
            "g/chi_obj": 10,
        }
        other_frequencies = {
            "a/chi_nsubj": 100,
            "b/chi_nsubj": 100,
            "c/chi_nsubj": 100,
            "d/chi_obj": 1,
            "e/chi_obj": 80,
            "f/chi_obj": 70,
            "g/chi_obj": 49,
            "other_only/chi_nsubj": 500,
        }

        anchors = _select_dependency_anchors(
            base_vectors=base,
            other_vectors=other,
            base_frequencies=base_frequencies,
            other_frequencies=other_frequencies,
            top_k_anchor=10,
            min_anchor_count=2,
        )

        self.assertEqual(
            anchors,
            [
                "a/chi_nsubj",
                "b/chi_nsubj",
                "c/chi_nsubj",
                "e/chi_obj",
                "f/chi_obj",
            ],
        )

        anchors_with_minimum = _select_dependency_anchors(
            base_vectors=base,
            other_vectors=other,
            base_frequencies=base_frequencies,
            other_frequencies=other_frequencies,
            top_k_anchor=1,
            min_anchor_count=6,
        )
        self.assertEqual(len(anchors_with_minimum), 6)

        with self.assertWarnsRegex(UserWarning, "only 7.*minimum.*10"):
            all_anchors = _select_dependency_anchors(
                base_vectors=base,
                other_vectors=other,
                base_frequencies=base_frequencies,
                other_frequencies=other_frequencies,
                top_k_anchor=1,
                min_anchor_count=10,
            )
        self.assertEqual(len(all_anchors), 7)

    def test_dependency_folder_uses_one_global_rotation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output_root = root / "aligned"
            words = ["a/chi_nsubj", "b/chi_nsubj", "c/chi_obj", "d/chi_obj"]
            period_vectors = {
                "t": [[1, 0], [0, 1], [-1, 0], [0, -1]],
                "t1": [[0, 1], [-1, 0], [0, -1], [1, 0]],
                "t2": [[-1, 0], [0, -1], [1, 0], [0, 1]],
            }
            for period, vectors in period_vectors.items():
                period_dir = input_root / period
                period_dir.mkdir(parents=True)
                keyed_vectors = KeyedVectors(vector_size=2)
                keyed_vectors.add_vectors(words, np.array(vectors, dtype=np.float32))
                keyed_vectors.save_word2vec_format(str(period_dir / f"{period}.txt"))
                (period_dir / "vocab.txt").write_text(
                    "".join(f"{word}\t100\n" for word in words),
                    encoding="utf-8",
                )

            results = align_depw2v_raw_vec_folder(
                input_root,
                output_root,
                periods=["t", "t1", "t2"],
                vocab_freq_filename="vocab.txt",
                top_k_anchor=1,
                min_anchor_count=4,
                save_formats=("keyed_vectors",),
            )

            self.assertEqual([result.anchor_count for result in results], [0, 4, 4])
            base = KeyedVectors.load(str(results[0].output_path))
            last = KeyedVectors.load(str(results[2].output_path))
            np.testing.assert_allclose(last.vectors, base.vectors, atol=1e-6)

    def test_region_specific_mode_applies_one_rotation_per_relation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_root = root / "input"
            output_root = root / "aligned"
            words = [
                "a/chi_nsubj",
                "b/chi_nsubj",
                "c/chi_nsubj",
                "a/chi_obj",
                "b/chi_obj",
                "c/chi_obj",
            ]
            base_vectors = np.array(
                [[1, 0], [0, 1], [1, 1], [1, 0], [0, 1], [1, 1]],
                dtype=np.float32,
            )
            other_vectors = np.array(
                [[0, 1], [-1, 0], [-1, 1], [0, -1], [1, 0], [1, -1]],
                dtype=np.float32,
            )
            for period, vectors in (("t", base_vectors), ("t1", other_vectors)):
                period_dir = input_root / period
                period_dir.mkdir(parents=True)
                keyed_vectors = KeyedVectors(vector_size=2)
                keyed_vectors.add_vectors(words, vectors)
                keyed_vectors.save_word2vec_format(str(period_dir / f"{period}.txt"))
                (period_dir / "vocab.txt").write_text(
                    "".join(
                        f"{word}\t{frequency}\n"
                        for word, frequency in zip(words, (100, 90, 1, 100, 90, 1))
                    ),
                    encoding="utf-8",
                )

            results = align_depw2v_raw_vec_folder(
                input_root,
                output_root,
                periods=["t", "t1"],
                vocab_freq_filename="vocab.txt",
                top_k_anchor_pct=25,
                min_anchor_count=2,
                alignment_mode="region_specific",
                save_formats=("keyed_vectors",),
            )

            self.assertEqual([result.anchor_count for result in results], [0, 4])
            base = KeyedVectors.load(str(results[0].output_path))
            aligned = KeyedVectors.load(str(results[1].output_path))
            np.testing.assert_allclose(aligned.vectors, base.vectors, atol=1e-6)

    def test_region_specific_mode_uses_minimum_per_relation(self) -> None:
        base = KeyedVectors(vector_size=2)
        other = KeyedVectors(vector_size=2)
        words = [
            "a/chi_nsubj",
            "b/chi_nsubj",
            "c/chi_nsubj",
            "d/chi_nsubj",
            "a/chi_obj",
        ]
        vectors = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]],
            dtype=np.float32,
        )
        base.add_vectors(words, vectors)
        other.add_vectors(words, vectors)
        frequencies = {word: 10 for word in words}
        with self.assertWarnsRegex(
            UserWarning,
            "Dependency regions, relation 'chi_obj'.*only 1.*minimum.*3",
        ):
            anchors_by_relation = _select_dependency_anchors_by_relation(
                base_vectors=base,
                other_vectors=other,
                base_frequencies=frequencies,
                other_frequencies=frequencies,
                top_k_anchor_pct=25,
                min_anchor_count=3,
            )
        self.assertEqual(
            {relation: len(anchors) for relation, anchors in anchors_by_relation.items()},
            {"chi_nsubj": 3, "chi_obj": 1},
        )

        _, anchor_count = _dependency_region_rotations(
            base_vectors=base,
            other_vectors=other,
            anchors_by_relation=anchors_by_relation,
        )
        self.assertEqual(anchor_count, 4)

    def test_region_specific_mode_requires_valid_percentage(self) -> None:
        for percentage in (None, 0, 101):
            with self.subTest(percentage=percentage):
                with self.assertRaisesRegex(ValueError, "top_k_anchor_pct"):
                    align_depw2v_raw_vec_folder(
                        "unused",
                        "unused",
                        periods=["t"],
                        vocab_freq_filename="vocab.txt",
                        top_k_anchor_pct=percentage,
                        alignment_mode="region_specific",
                    )

        with self.assertRaisesRegex(ValueError, "omitted in global mode"):
            align_depw2v_raw_vec_folder(
                "unused",
                "unused",
                periods=["t"],
                vocab_freq_filename="vocab.txt",
                top_k_anchor_pct=5,
                alignment_mode="global",
            )

    def test_dependency_item_requires_relation_suffix(self) -> None:
        vectors = KeyedVectors(vector_size=1)
        vectors.add_vectors(["missing_relation"], np.array([[1]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "item/relation"):
            _select_dependency_anchors(
                base_vectors=vectors,
                other_vectors=vectors,
                base_frequencies={"missing_relation": 10},
                other_frequencies={"missing_relation": 10},
                top_k_anchor=1,
                min_anchor_count=None,
            )


if __name__ == "__main__":
    unittest.main()
