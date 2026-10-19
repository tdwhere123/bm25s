import json
import tempfile
import unittest
import unittest.mock

import bm25s
from bm25s.utils import json_functions


class TestSaveUnserializableCorpus(unittest.TestCase):
    """save() must raise instead of silently dropping corpus entries and misaligning them with the index."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = temporary.name
        self.texts = ["a cat purrs", "a dog barks", "a fish swims"]
        tokens = bm25s.tokenize(self.texts, show_progress=False)
        self.model = bm25s.BM25(backend="numpy")
        self.model.index(tokens, show_progress=False)
        self.corpus = [
            {"id": 0, "text": self.texts[0]},
            {"id": 1, "text": self.texts[1], "tags": {"pet"}},
            {"id": 2, "text": self.texts[2]},
        ]

    def assert_save_raises_type_error(self, mmap=False):
        try:
            json_functions.dumps(self.corpus[1], ensure_ascii=False)
        except Exception as e:
            encoder_error = e
        else:
            self.fail("corpus entry 1 should not be JSON serializable")

        try:
            self.model.save(self.directory, corpus=self.corpus, show_progress=False)
        except TypeError as e:
            self.assertIn("Document at index 1", str(e))
            self.assertIn(str(encoder_error), str(e))
            self.assertIsNotNone(e.__cause__)
            self.assertIsInstance(e.__cause__, TypeError)
            self.assertIs(type(e.__cause__), type(encoder_error))
            return

        loaded = bm25s.BM25.load(
            self.directory, load_corpus=True, mmap=mmap, show_progress=False
        )
        if mmap:
            self.addCleanup(loaded.corpus.close)
        results, _ = loaded.retrieve([["dog"]], k=1, show_progress=False)
        self.fail(
            f"save() did not raise TypeError, and searching 'dog' after reload "
            f"(mmap={mmap}) returned {results[0, 0]!r}"
        )

    def test_unserializable_entry(self):
        self.assert_save_raises_type_error()

    def test_unserializable_entry_mmap(self):
        self.assert_save_raises_type_error(mmap=True)

    @unittest.mock.patch("bm25s.utils.json_functions.dumps", json_functions.dumps_with_builtin)
    @unittest.mock.patch("bm25s.utils.json_functions.loads", json.loads)
    def test_unserializable_entry_no_orjson(self):
        self.assert_save_raises_type_error()


if __name__ == "__main__":
    unittest.main()
