"""A small BM25 ranker for keyword search over chunk texts."""

import math
import re
from collections import Counter
from collections.abc import Sequence

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and split it into alphanumeric/underscore runs.

    ``requires_grad`` stays one token while ``torch.nn.Module`` becomes ``torch``, ``nn``
    and ``module``. There is no stemming or stop-word removal; BM25's IDF already
    down-weights common words.
    """
    return _TOKEN.findall(text.lower())


class BM25:
    """Standard BM25 over a fixed list of documents.

    Uses the non-negative IDF ``log(1 + (N - df + 0.5) / (df + 0.5))``, so a term that
    appears in most documents contributes little instead of a negative score.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._term_counts = [Counter(tokenize(d)) for d in documents]
        self._lengths = [sum(c.values()) for c in self._term_counts]
        self._avg_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        doc_freq: Counter[str] = Counter()
        for counts in self._term_counts:
            doc_freq.update(counts.keys())
        n = len(self._term_counts)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> list[float]:
        """Return the BM25 score of every document for ``query``, in document order.

        A document sharing no term with the query scores 0.
        """
        terms = set(tokenize(query)) & self._idf.keys()
        out = []
        for counts, length in zip(self._term_counts, self._lengths, strict=True):
            norm = 1 - self._b + self._b * length / self._avg_length if self._avg_length else 1
            score = 0.0
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self._idf[term] * tf * (self._k1 + 1) / (tf + self._k1 * norm)
            out.append(score)
        return out
