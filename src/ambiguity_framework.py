# -*- coding: utf-8 -*-
"""
ambiguity_framework.py
-----------------------
Implementation of the proposed framework for "Identifying Inter and Intra
Domain-Specific Ambiguity (Polysemous Words) from Natural Language Software
Requirements".

This module implements, in code, the seven-stage pipeline described in the
project report (Section 3, Figure 3.1 / 3.4) and Algorithm 3.1, updated
and grounded per formal review recommendations:

    1. NLP Preprocessing            -> Preprocessor
    2. Candidate Polysemous Word
       Detection                    -> CandidateDetector
    3. Context Analysis &
       Domain Identification        -> DomainIdentifier (Lexicographer + Descriptors)
    4. Context Representation       -> ContextRepresenter (TF-IDF / S-BERT)
    5. Ambiguity Score computation  -> AmbiguityScorer (Spread + Baseline Polysemy + Domain Diversity)
    6. Inter/Intra-Domain
       Classification               -> classify_ambiguity() (over competing senses Sigma*)
    7. Final Ambiguity Report       -> AmbiguityPipeline.analyse_sentence()

Formalism updates (Addressing Review Item 4 & 5):
    - Redefines candidate senses T_k(w, c) and competing sense set Sigma*(w, c)
      using contextual proximity margin gamma.
    - Renames "Contextual Confidence" to "Baseline Polysemy" (P_0) reflecting
      lexical polysemy degree.
    - Genuinely integrates Domain Relevance / Diversity (Div) into the Ambiguity
      Score: Score = w1 * Spread + w2 * Polysemy + w3 * DomainDiversity.
    - Supports learning or tuning feature weights via Logistic Regression / Dev split.
    - Integrates WordNet lexicographer categories (synset.lexname()) into DomainIdentifier.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import spacy
import nltk
try:
    from nltk.corpus import wordnet as wn
    wn.synsets("test")
except (LookupError, AttributeError):
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

from domain_glossary import (
    DOMAIN_DESCRIPTORS,
    CANDIDATE_GLOSSARY_TERMS,
    LEXNAME_TO_DOMAIN_PRIORS,
)


def _enriched_gloss_text(syn) -> str:
    """Builds a richer bag-of-words representation of a WordNet sense than
    the bare gloss, by appending WordNet usage examples, the sense's own
    lemma names, and its direct hypernyms' definitions. This substantially
    increases vocabulary overlap with real requirement sentences and domain
    descriptors.
    """
    parts = [syn.definition()]
    parts.extend(syn.examples())
    parts.extend(name.replace("_", " ") for name in syn.lemma_names())
    for hyper in syn.hypernyms():
        parts.append(hyper.definition())
    return ". ".join(parts)


# --------------------------------------------------------------------------
# Stage 1: NLP Preprocessing  (report Section 3.2, Figure 3.2)
# --------------------------------------------------------------------------
class Preprocessor:
    """Sentence segmentation, tokenisation, lemmatisation, POS tagging and
    dependency parsing, using spaCy's English pipeline.
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)

    def process(self, text: str):
        return self.nlp(text)


# --------------------------------------------------------------------------
# Stage 2: Candidate Polysemous Word Detection (report Section 3.3, Fig 3.3)
# --------------------------------------------------------------------------
@dataclass
class CandidateWord:
    lemma: str
    pos_wn: str            # WordNet POS tag used to look up senses
    token_index: int       # position of the token within the sentence
    surface_form: str
    senses: List["wn.synset"] = field(default_factory=list)


class CandidateDetector:
    """Identifies candidate polysemous words using WordNet sense counts and
    the domain glossary, filtering out functional stop words.
    """

    _POS_MAP = {"NOUN": wn.NOUN, "PROPN": wn.NOUN, "VERB": wn.VERB}

    def __init__(self, min_wordnet_senses: int = 3, min_word_len: int = 3, use_glossary: bool = True):
        self.min_wordnet_senses = min_wordnet_senses
        self.min_word_len = min_word_len
        self.use_glossary = use_glossary

    def detect(self, doc) -> List[CandidateWord]:
        candidates: List[CandidateWord] = []
        seen_positions = set()
        for tok in doc:
            if tok.i in seen_positions:
                continue
            if tok.is_stop or tok.is_punct or not tok.is_alpha:
                continue
            if len(tok.lemma_) < self.min_word_len:
                continue
            wn_pos = self._POS_MAP.get(tok.pos_)
            if wn_pos is None:
                continue

            lemma = tok.lemma_.lower()
            senses = wn.synsets(lemma, pos=wn_pos)

            is_glossary_term = self.use_glossary and (lemma in CANDIDATE_GLOSSARY_TERMS)
            is_wordnet_polysemous = len(senses) >= self.min_wordnet_senses

            if is_glossary_term or is_wordnet_polysemous:
                if is_glossary_term and len(senses) < 2:
                    senses = wn.synsets(lemma)
                candidates.append(
                    CandidateWord(
                        lemma=lemma,
                        pos_wn=wn_pos,
                        token_index=tok.i,
                        surface_form=tok.text,
                        senses=senses,
                    )
                )
                seen_positions.add(tok.i)
        return candidates


# --------------------------------------------------------------------------
# Stage 3 & 4: Context Representation and Domain Identification
# --------------------------------------------------------------------------
class ContextRepresenter:
    """Builds a shared vector space over:
        - WordNet sense definitions (glosses),
        - domain descriptor texts (DOMAIN_DESCRIPTORS), and
        - requirement sentences.
    """

    def __init__(self):
        self.vectoriser = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z]{2,}\b",
        )
        self._fitted = False

    def fit(self, corpus: List[str]):
        self.vectoriser.fit(corpus)
        self._fitted = True
        self.domain_names = list(DOMAIN_DESCRIPTORS.keys())
        self.domain_vectors = self.vectoriser.transform(
            [DOMAIN_DESCRIPTORS[d] for d in self.domain_names]
        )
        return self

    def transform(self, texts: List[str]):
        assert self._fitted, "ContextRepresenter must be fit() before use."
        return self.vectoriser.transform(texts)

    def best_domain(self, text_vector, synset: Optional[wn.synset] = None) -> Tuple[str, float]:
        """Return (domain_name, combined_similarity) incorporating both descriptor
        similarity and WordNet lexicographer file priors (synset.lexname()).
        """
        sims = cosine_similarity(text_vector, self.domain_vectors)[0]
        domain_scores = {dom: float(sims[i]) for i, dom in enumerate(self.domain_names)}

        # Incorporate WordNet lexicographer category prior if synset provided
        if synset is not None:
            lexname = synset.lexname()
            priors = LEXNAME_TO_DOMAIN_PRIORS.get(lexname, {})
            for dom, prior_weight in priors.items():
                if dom in domain_scores:
                    domain_scores[dom] = 0.70 * domain_scores[dom] + 0.30 * prior_weight

        best_dom = max(domain_scores, key=domain_scores.get)
        return best_dom, domain_scores[best_dom]


class DomainIdentifier:
    """Answers which application domain a WordNet sense or requirement sentence belongs to."""

    def __init__(self, representer: ContextRepresenter):
        self.representer = representer

    def domain_of_sentence(self, sentence: str) -> Tuple[str, float]:
        vec = self.representer.transform([sentence])
        return self.representer.best_domain(vec)

    def domain_of_sense(self, synset: wn.synset) -> Tuple[str, float]:
        gloss_text = _enriched_gloss_text(synset)
        vec = self.representer.transform([gloss_text])
        return self.representer.best_domain(vec, synset=synset)


# --------------------------------------------------------------------------
# Stage 5: Ambiguity Score (Formalism matching Implementation)
# --------------------------------------------------------------------------
@dataclass
class SenseScore:
    synset_name: str
    gloss: str
    domain: str
    domain_similarity: float
    context_similarity: float


@dataclass
class AmbiguityResult:
    word: str
    surface_form: str
    sentence: str
    sentence_domain: str
    sentence_domain_similarity: float
    ranked_senses: List[SenseScore]
    semantic_similarity_spread: float
    baseline_polysemy: float             # Formally renamed from contextual_confidence
    domain_diversity: float              # Div
    ambiguity_score: float
    is_ambiguous: bool
    ambiguity_type: str                  # "Inter-Domain" | "Intra-Domain" | "None"
    explanation: str

    @property
    def contextual_confidence(self) -> float:
        """Backward-compatible alias for baseline_polysemy."""
        return self.baseline_polysemy


class AmbiguityScorer:
    """Computes the Ambiguity Score integrating three distinct factors:
        1. Semantic Similarity Spread (Spread): Relative proximity of top candidate senses to context.
        2. Baseline Polysemy (P_0): Lexical breadth / number of senses available.
        3. Domain Diversity (Div): Dispersion of candidate senses across distinct application domains.

    Score(w, c) = w1 * Spread + w2 * P_0 + w3 * Div
    where weights w1, w2, w3 >= 0 and sum to 1.0 (empirically tuned / learned on Dev).
    """

    def __init__(
        self,
        threshold: float = 0.35,
        top_k: int = 3,
        gamma_margin: float = 0.08,
        w_spread: float = 0.45,
        w_polysemy: float = 0.35,
        w_diversity: float = 0.20,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self.gamma_margin = gamma_margin
        self.w_spread = w_spread
        self.w_polysemy = w_polysemy
        self.w_diversity = w_diversity

    def fit_weights(self, X_features: np.ndarray, y_labels: np.ndarray):
        """Learn feature weights using logistic regression on development set."""
        lr = LogisticRegression(fit_intercept=False, C=1.0)
        lr.fit(X_features, y_labels)
        coefs = np.maximum(0.01, lr.coef_[0])
        norm_coefs = coefs / np.sum(coefs)
        self.w_spread = float(norm_coefs[0])
        self.w_polysemy = float(norm_coefs[1])
        self.w_diversity = float(norm_coefs[2])
        return self

    def score(
        self,
        context_vector,
        candidate: CandidateWord,
        representer: ContextRepresenter,
        domain_identifier: DomainIdentifier,
    ) -> Tuple[List[SenseScore], float, float, float, float, List[SenseScore]]:

        sense_scores: List[SenseScore] = []
        for syn in candidate.senses:
            gloss = syn.definition()
            gloss_vec = representer.transform([_enriched_gloss_text(syn)])
            context_sim = float(cosine_similarity(context_vector, gloss_vec)[0][0])
            domain, domain_sim = domain_identifier.domain_of_sense(syn)
            sense_scores.append(
                SenseScore(
                    synset_name=syn.name(),
                    gloss=gloss,
                    domain=domain,
                    domain_similarity=domain_sim,
                    context_similarity=context_sim,
                )
            )

        # Rank senses by context similarity: T_k(w, c)
        sense_scores.sort(key=lambda s: s.context_similarity, reverse=True)
        top = sense_scores[: self.top_k] if sense_scores else []

        # 1. Semantic Similarity Spread (Spread)
        _MIN_MEANINGFUL_SIM = 0.04
        if len(top) >= 2 and top[0].context_similarity >= _MIN_MEANINGFUL_SIM:
            s1, s2 = top[0].context_similarity, top[1].context_similarity
            relative_gap = (s1 - s2) / max(0.001, (s1 + s2))
            semantic_similarity_spread = float(np.clip(1.0 - relative_gap, 0.0, 1.0))
        else:
            semantic_similarity_spread = 0.50

        # 2. Baseline Polysemy (P_0) - formally renamed from contextual confidence
        n_senses = len(candidate.senses)
        baseline_polysemy = float(np.clip((n_senses - 2) / 10.0, 0.0, 1.0))

        # 3. Domain Relevance / Diversity (Div)
        distinct_domains = {s.domain for s in top}
        domain_diversity = float((len(distinct_domains) - 1) / max(1, (len(top) - 1))) if len(top) > 1 else 0.0

        # Formally integrated Ambiguity Score (incorporating Div as claimed in Abstract)
        ambiguity_score = float(
            self.w_spread * semantic_similarity_spread
            + self.w_polysemy * baseline_polysemy
            + self.w_diversity * domain_diversity
        )

        # 4. Competing senses Sigma*(w, c) within gamma margin of top sense
        sigma_star: List[SenseScore] = []
        if top:
            top_sim = top[0].context_similarity
            sigma_star = [s for s in top if (top_sim - s.context_similarity) <= self.gamma_margin]
            if len(sigma_star) < 2 and len(top) >= 2:
                sigma_star = top[:2]

        return sense_scores, semantic_similarity_spread, baseline_polysemy, domain_diversity, ambiguity_score, sigma_star


# --------------------------------------------------------------------------
# Stage 6: Inter / Intra-Domain Classification
# --------------------------------------------------------------------------
def classify_ambiguity(competing_senses: List[SenseScore]) -> str:
    """Classify ambiguity as Inter-Domain or Intra-Domain across the competing
    senses Sigma*(w, c):
        - Inter-Domain: competing senses span different application domains.
        - Intra-Domain: competing senses belong to the same application domain
          yet denote distinct concepts/functions within it.
    """
    if len(competing_senses) < 2:
        return "None"
    domains = {s.domain for s in competing_senses[:2]}
    return "Inter-Domain" if len(domains) > 1 else "Intra-Domain"


def build_explanation(result_word: str, sentence_domain: str, top: List[SenseScore], ambiguity_type: str) -> str:
    if not top:
        return f'"{result_word}" has no WordNet senses available for analysis.'
    lines = [f'The word "{result_word}" appears in a sentence whose detected domain is "{sentence_domain}".']
    for i, s in enumerate(top[:2], start=1):
        lines.append(
            f"  Candidate sense {i}: [{s.domain}] {s.gloss} "
            f"(context similarity={s.context_similarity:.2f})"
        )
    if ambiguity_type == "Inter-Domain":
        lines.append(
            f'Classified as INTER-DOMAIN ambiguity: competing senses belong to '
            f'different application domains ("{top[0].domain}" vs. "{top[1].domain}").'
        )
    elif ambiguity_type == "Intra-Domain":
        lines.append(
            f'Classified as INTRA-DOMAIN ambiguity: competing senses belong to '
            f'the same "{top[0].domain}" domain but represent distinct concepts/operations.'
        )
    else:
        lines.append("Not flagged as ambiguous in this context.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Stage 7: Orchestration Pipeline (Algorithm 3.1)
# --------------------------------------------------------------------------
class AmbiguityPipeline:
    """End-to-end orchestration of Algorithm 3.1."""

    def __init__(
        self,
        min_wordnet_senses: int = 3,
        min_word_len: int = 3,
        ambiguity_threshold: float = 0.35,
        top_k_senses: int = 3,
        use_glossary: bool = True,
        w_spread: float = 0.45,
        w_polysemy: float = 0.35,
        w_diversity: float = 0.20,
    ):
        self.preprocessor = Preprocessor()
        self.candidate_detector = CandidateDetector(min_wordnet_senses, min_word_len, use_glossary=use_glossary)
        self.representer = ContextRepresenter()
        self.domain_identifier = DomainIdentifier(self.representer)
        self.scorer = AmbiguityScorer(
            threshold=ambiguity_threshold,
            top_k=top_k_senses,
            w_spread=w_spread,
            w_polysemy=w_polysemy,
            w_diversity=w_diversity,
        )
        self._fitted = False

    def fit_vocabulary(self, sentences: List[str]):
        gloss_corpus = set()
        for sent in sentences:
            doc = self.preprocessor.process(sent)
            for cand in self.candidate_detector.detect(doc):
                for syn in cand.senses:
                    gloss_corpus.add(_enriched_gloss_text(syn))

        corpus = list(sentences) + list(gloss_corpus) + list(DOMAIN_DESCRIPTORS.values())
        self.representer.fit(corpus)
        self._fitted = True
        return self

    def analyse_sentence(self, sentence: str) -> List[AmbiguityResult]:
        assert self._fitted, "Call fit_vocabulary() before analyse_sentence()."

        doc = self.preprocessor.process(sentence)
        candidates = self.candidate_detector.detect(doc)
        sentence_domain, sentence_domain_sim = self.domain_identifier.domain_of_sentence(sentence)
        context_vector = self.representer.transform([sentence])

        results: List[AmbiguityResult] = []
        for cand in candidates:
            (
                sense_scores,
                spread,
                baseline_poly,
                diversity,
                ambiguity_score,
                sigma_star,
            ) = self.scorer.score(context_vector, cand, self.representer, self.domain_identifier)

            top = sorted(sense_scores, key=lambda s: s.context_similarity, reverse=True)[:3]
            is_ambiguous = ambiguity_score >= self.scorer.threshold and len(top) >= 2
            ambiguity_type = classify_ambiguity(sigma_star) if is_ambiguous else "None"

            explanation = build_explanation(cand.lemma, sentence_domain, top, ambiguity_type)

            results.append(
                AmbiguityResult(
                    word=cand.lemma,
                    surface_form=cand.surface_form,
                    sentence=sentence,
                    sentence_domain=sentence_domain,
                    sentence_domain_similarity=sentence_domain_sim,
                    ranked_senses=top,
                    semantic_similarity_spread=spread,
                    baseline_polysemy=baseline_poly,
                    domain_diversity=diversity,
                    ambiguity_score=ambiguity_score,
                    is_ambiguous=is_ambiguous,
                    ambiguity_type=ambiguity_type,
                    explanation=explanation,
                )
            )
        return results

    def analyse_corpus(self, sentences: List[str]) -> List[AmbiguityResult]:
        all_results = []
        for sent in sentences:
            all_results.extend(self.analyse_sentence(sent))
        return all_results
