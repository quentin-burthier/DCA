from data.token_indexer import TokenIndexer

class TestTokenIndex:

    def test_in_voc(self):
        text = "This is a test .".split(' ')
        word2id = {
            "This": 1,
            "is": 2,
            "a": 3,
            "test": 4,
            ".": 5
        }
        token_indexer = TokenIndexer(word2id, 0)
        indexed = [int(idx) for idx in token_indexer(text)]
        assert indexed == [1, 2, 3, 4, 5], indexed

    def test_oov(self):
        text = "This is another test .".split(' ')
        word2id = {
            "This": 1,
            "is": 2,
            "a": 3,
            "test": 4,
            ".": 5
        }
        token_indexer = TokenIndexer(word2id, 0)
        indexed = [int(idx) for idx in token_indexer(text)]
        assert indexed == [1, 2, 0, 4, 5], indexed
