from news_crawler.processors.summarizer import _chunk_text


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return "".join(chr(token_id) for token_id in token_ids)


def test_chunk_text_splits_on_multiple_sentence_endings():
    tokenizer = _FakeTokenizer()
    text = "Alpha! Beta? Gamma."

    chunks = _chunk_text(text, tokenizer, max_tokens=10)

    assert chunks == ["Alpha!", "Beta?", "Gamma."]


def test_chunk_text_hard_splits_oversized_sentence():
    tokenizer = _FakeTokenizer()
    text = "abcdefghijklmno"

    chunks = _chunk_text(text, tokenizer, max_tokens=5)

    assert chunks == ["abcde", "fghij", "klmno"]
    assert all(len(tokenizer.encode(chunk, add_special_tokens=False)) <= 5 for chunk in chunks)
