from data.token_indexer import TokenIndexer


def test_1():
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

def test_2():
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
