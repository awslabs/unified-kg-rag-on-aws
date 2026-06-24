# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Any, ClassVar, TypeAlias

from datasketch import MinHash, MinHashLSH

from aws_graphrag.domain.models import Config, ResolutionMethod
from aws_graphrag.shared import get_logger
from aws_graphrag.shared.utils import default_max_workers, normalize_name

logger = get_logger(__name__)

MatchResult: TypeAlias = tuple[str, float] | None


class FuzzyMatcher:
    STOPWORDS: ClassVar[set[str]] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
    }

    _RE_NON_WORD_CHARS = re.compile(r"[^\w\s-]")
    _RE_EXTRA_SPACES = re.compile(r"\s+")
    _RE_WORD_BOUNDARY = re.compile(r"\b\w+\b")
    _RE_DIGITS = re.compile(r"\d+")

    def __init__(
        self,
        candidates: list[str],
        resolution_method: str = ResolutionMethod.SEQUENCE_MATCHER,
        similarity_threshold: float = 0.5,
        minhash_permutations: int = 128,
        include_partial_matching: bool = False,
    ):
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")

        self.candidates = list(set(candidates))
        self.resolution_method = resolution_method
        self.similarity_threshold = similarity_threshold
        self.minhash_permutations = minhash_permutations
        self.include_partial_matching = include_partial_matching

        logger.debug(
            "Initializing FuzzyMatcher with %s candidates, method: %s, threshold: %s",
            len(self.candidates),
            resolution_method,
            similarity_threshold,
        )

        self._build_indices()
        if self.resolution_method == ResolutionMethod.MINHASH:
            self._build_minhash_lsh()

    def _build_indices(self) -> None:
        self.exact_index = {name: name for name in self.candidates}
        self.normalized_index = {normalize_name(name): name for name in self.candidates}
        self.abbreviation_index: dict[str, str] = {}
        for name in self.candidates:
            for abbrev in self._generate_abbreviations(name):
                if abbrev not in self.abbreviation_index:
                    self.abbreviation_index[abbrev] = name

        logger.debug(
            "Built indices: %s exact, %s normalized, %s abbreviations",
            len(self.exact_index),
            len(self.normalized_index),
            len(self.abbreviation_index),
        )

    def _build_minhash_lsh(self) -> None:
        self.lsh = MinHashLSH(
            threshold=self.similarity_threshold, num_perm=self.minhash_permutations
        )
        self.minhashes = {}
        for name in self.candidates:
            minhash = self._create_minhash(name, self.minhash_permutations)
            self.minhashes[name] = minhash
            self.lsh.insert(name, minhash)

        logger.debug("Built MinHash LSH with %s candidates", len(self.minhashes))

    @classmethod
    def _create_minhash(
        cls, text: str, minhash_permutations: int = 128, n_grams: int = 3
    ) -> MinHash:
        minhash = MinHash(num_perm=minhash_permutations)
        normalized_text = normalize_name(text)

        if len(normalized_text) < n_grams:
            shingles = {normalized_text}
        else:
            shingles = {
                normalized_text[i : i + n_grams]
                for i in range(len(normalized_text) - n_grams + 1)
            }

        for shingle in shingles:
            minhash.update(shingle.encode("utf8"))
        return minhash

    @classmethod
    def _generate_abbreviations(cls, text: str) -> set[str]:
        abbrevs: set[str] = set()
        if not text:
            return abbrevs

        # Acronyms (first-letter-of-each-word) and caps-extraction are Latin/
        # cased-script notions; for scripts without case or word spacing (CJK,
        # etc.) they produce noise, so only generate them when the text actually
        # contains ASCII letters. Non-Latin names rely on normalized + token /
        # sequence similarity instead.
        if not any("a" <= c.lower() <= "z" for c in text):
            return abbrevs

        words = [w for w in cls._RE_WORD_BOUNDARY.findall(text.lower()) if w]
        if len(words) > 1:
            acronym = "".join(w[0] for w in words).upper()
            if len(acronym) > 1:
                abbrevs.add(acronym)

        caps = "".join(c for c in text if c.isupper())
        if len(caps) > 1:
            abbrevs.add(caps)

        return abbrevs

    def find_all_matches(self, query: str) -> list[tuple[str, float]]:
        """Return ALL candidates similar to ``query`` (used for entity grouping).

        Unlike :meth:`find_best_match`, this intentionally omits the abbreviation
        tier: grouping fans every match into a merge cluster, and abbreviation
        matches ("AC" -> "Acme Corp") are too aggressive there (they would
        over-merge unrelated entities). Exact/normalized equality is still
        covered — ``_create_minhash`` normalizes (NFKC/casefold) before
        shingling, so normalized-equal names (incl. CJK) score 1.0 under LSH.
        """
        if self.resolution_method == ResolutionMethod.MINHASH:
            return self._find_all_lsh_matches(query)
        return self._find_all_string_similarity_matches(query)

    def _find_all_lsh_matches(self, query: str) -> list[tuple[str, float]]:
        if self.resolution_method != ResolutionMethod.MINHASH:
            return []

        query_minhash = self._create_minhash(query, self.minhash_permutations)
        similar_candidates = self.lsh.query(query_minhash)
        if not similar_candidates:
            return []

        matches = [
            (str(candidate), query_minhash.jaccard(self.minhashes[str(candidate)]))
            for candidate in similar_candidates
        ]
        return [match for match in matches if match[1] >= self.similarity_threshold]

    def _find_all_string_similarity_matches(
        self, query: str
    ) -> list[tuple[str, float]]:
        normalized_query = normalize_name(query)
        matches = []

        for candidate_name in self.candidates:
            if query == candidate_name:
                continue

            normalized_candidate = normalize_name(candidate_name)
            score = SequenceMatcher(
                None, normalized_query, normalized_candidate
            ).ratio()
            if score >= self.similarity_threshold:
                matches.append((candidate_name, score))

        return matches

    def find_best_match(self, query: str) -> MatchResult:
        if not query or not query.strip():
            return None

        if match := self.exact_index.get(query):
            logger.debug("Found exact match for '%s': %s", query, match)
            return match, 1.0

        normalized_query = normalize_name(query)
        if match := self.normalized_index.get(normalized_query):
            logger.debug("Found normalized match for '%s': %s", query, match)
            return match, 0.95

        for abbrev in self._generate_abbreviations(query):
            if match := self.abbreviation_index.get(abbrev):
                logger.debug(
                    "Found abbreviation match for '%s': %s (via '%s')",
                    query,
                    match,
                    abbrev,
                )
                return match, 0.90

        result = self._find_best_fuzzy_match(query, normalized_query)
        if result:
            logger.debug(
                "Found fuzzy match for '%s': %s (score: %.3f)",
                query,
                result[0],
                result[1],
            )
        else:
            logger.debug("No match found for '%s'", query)

        return result

    def _find_best_fuzzy_match(self, query: str, normalized_query: str) -> MatchResult:
        matches = []

        if match_result := self._find_jaccard_match(query, False):
            matches.append((match_result[0], match_result[1] * 0.9))

        if match_result := self._find_jaccard_match(query, True):
            matches.append((match_result[0], match_result[1] * 0.85))

        if self.include_partial_matching:
            if match_result := self._find_partial_match(normalized_query):
                matches.append((match_result[0], match_result[1] * 0.8))

        if self.resolution_method == ResolutionMethod.MINHASH:
            if match_result := self._find_minhash_match(query):
                matches.append((match_result[0], match_result[1] * 0.75))
        else:
            if match_result := self._find_string_similarity_match(normalized_query):
                matches.append((match_result[0], match_result[1] * 0.75))

        if not matches:
            return None

        best_candidate, best_score = max(matches, key=lambda item: item[1])
        return (
            (best_candidate, best_score)
            if best_score >= self.similarity_threshold
            else None
        )

    def _find_jaccard_match(self, query: str, meaningful: bool) -> MatchResult:
        query_tokens = self._extract_tokens(query, meaningful_only=meaningful)
        if not query_tokens:
            return None

        best_candidate = None
        best_score = 0.0

        for candidate_name in self.candidates:
            candidate_tokens = self._extract_tokens(
                candidate_name, meaningful_only=meaningful
            )
            if not candidate_tokens:
                continue

            intersection = len(query_tokens.intersection(candidate_tokens))
            union = len(query_tokens.union(candidate_tokens))
            score = intersection / union if union > 0 else 0.0

            if score > best_score:
                best_score = score
                best_candidate = candidate_name

        return (
            (best_candidate, best_score) if best_candidate and best_score > 0 else None
        )

    def _extract_tokens(self, text: str, meaningful_only: bool = False) -> set[str]:
        normalized = normalize_name(text)
        words = set(normalized.split())

        if meaningful_only:
            return {
                w
                for w in words
                if w not in self.STOPWORDS and len(w) > 1 and not w.isdigit()
            }

        tokens = {w for w in words if len(w) > 1}
        tokens.update(self._RE_DIGITS.findall(normalized))
        return tokens

    def _find_partial_match(self, normalized_query: str) -> MatchResult:
        best_match = None
        best_score = 0.0

        for candidate_name in self.candidates:
            normalized_candidate = normalize_name(candidate_name)
            score = 0.0

            if normalized_query in normalized_candidate:
                score = len(normalized_query) / len(normalized_candidate)
            elif normalized_candidate in normalized_query:
                score = len(normalized_candidate) / len(normalized_query)

            if score > best_score:
                best_score = score
                best_match = candidate_name

        return (best_match, best_score) if best_match and best_score > 0 else None

    def _find_minhash_match(self, query: str) -> MatchResult:
        query_minhash = self._create_minhash(query, self.minhash_permutations)
        similar_candidates = self.lsh.query(query_minhash)
        if not similar_candidates:
            return None

        matches = [
            (str(candidate), query_minhash.jaccard(self.minhashes[str(candidate)]))
            for candidate in similar_candidates
        ]
        return max(matches, key=lambda item: item[1]) if matches else None

    def _find_string_similarity_match(self, normalized_query: str) -> MatchResult:
        matches = [
            (
                candidate_name,
                SequenceMatcher(
                    None, normalized_query, normalize_name(candidate_name)
                ).ratio(),
            )
            for candidate_name in self.candidates
        ]
        return max(matches, key=lambda item: item[1]) if matches else None


class BaseResolver(ABC):
    def __init__(
        self,
        config: Config,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.config = config
        self.max_workers = max_workers or default_max_workers()
        self.use_process_pool = use_process_pool
        self.show_progress = show_progress

        logger.info(
            "Resolver initialized with %s workers, process pool: %s",
            self.max_workers,
            use_process_pool,
        )

    @abstractmethod
    def resolve(self, *args: Any, **kwargs: Any) -> Sequence[Any]:
        pass

    def _create_fuzzy_matcher(
        self, candidate_texts: list[str], **kwargs: Any
    ) -> FuzzyMatcher:
        logger.debug(
            "Creating FuzzyMatcher with %s candidates, method: '%s', threshold: %s",
            len(candidate_texts),
            self.config.processing.resolution_method.value,
            self.config.processing.similarity_threshold,
        )
        return FuzzyMatcher(
            candidate_texts,
            resolution_method=self.config.processing.resolution_method,
            similarity_threshold=self.config.processing.similarity_threshold,
            **kwargs,
        )

    @staticmethod
    def _get_most_common_value(values: list[str]) -> str:
        if not values:
            return ""
        non_empty_values = [v for v in values if v]
        if not non_empty_values:
            return ""
        return Counter(non_empty_values).most_common(1)[0][0]

    @staticmethod
    def _merge_attributes(attributes_list: list[dict[str, Any]]) -> dict[str, Any]:
        merged = {}
        for attrs in attributes_list:
            if attrs:
                merged.update(attrs)
        return merged

    @staticmethod
    def _merge_descriptions(descriptions: list[str]) -> str:
        valid_descriptions = [desc for desc in descriptions if desc and desc.strip()]
        return "; ".join(valid_descriptions) if valid_descriptions else ""

    @staticmethod
    def _merge_lists(lists: list[list[str]]) -> list[str]:
        merged_set = set()
        for lst in lists:
            if lst:
                merged_set.update(lst)
        return sorted(merged_set)
