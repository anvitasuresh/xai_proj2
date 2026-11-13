"""Rule-based dialect variants via Multi-VALUE (Ziems et al., ACL 2023), restricted to well-attested features.

Why rule-based: asking an LLM to rewrite text "in AAE" tends to produce caricature, which would test the
judge's reaction to a stereotype rather than to the dialect. Multi-VALUE implements grammatical features
from eWAVE, the electronic World Atlas of Varieties of English (Kortmann, Lunkenheimer & Ehret, 2020),
and records which ones fired. That makes every variant traceable feature by feature and lets stage 3 (E3)
ablate one feature at a time.

On top of Multi-VALUE we add four safeguards against caricature (see docs/ethics.md):
  1. Attestation filter: only features eWAVE rates "A" (pervasive or obligatory) for that variety
     (`min_attestation: 1.0`). Features rated B/C (less frequent or rare) are dropped.
  2. Blocklist (`exclude`): features removed with a stated reason in configs/data.yaml, e.g. ones that are
     ungrammatical in our message context or that Multi-VALUE maps to the wrong variety.
  3. Feature density (`feature_prob`): each allowed feature is switched on per message with this
     probability, so a message doesn't stack every feature at once (real speakers vary, and stacking
     reads as exaggerated). The features actually used are recorded.
  4. No untracked slang: Multi-VALUE's surface "contraction" pass rewrites e.g. "going to" -> "I'ma",
     "have to" -> "gotta", "trying to" -> "tryna" for *every* dialect, outside any eWAVE feature. We keep
     only its error-repair rules.

Each transformer returns (new_text, features_applied). Multi-VALUE is imported lazily because it needs
its own environment (pinned spaCy/stanza/nltk); see docs/design_decisions.md.
"""

from __future__ import annotations

import random
import re
from typing import Any, Protocol


class Transformer(Protocol):
    name: str

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]: ...


class IdentityTransformer:
    """MAE: no change. Present so every group goes through the same code path."""

    def __init__(self, name: str = "mae"):
        self.name = name

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]:
        return text, {}


def _skip_spacy_download_if_installed() -> None:
    """Multi-VALUE's __init__ calls spacy.cli.download("en_core_web_sm") on every import, which shells
    out to pip and races when several shards start at once. setup_local.sh already installs the model,
    so make that call a no-op when the model is present."""
    import spacy
    import spacy.cli

    if spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download = lambda *args, **kwargs: None


def _repair_only_contractions(string: str) -> str:
    """The error-repair half of Multi-VALUE's surface_contract, without its slang substitutions."""
    string = re.sub(r"\bnn't\b", "n't", string, flags=re.IGNORECASE)
    string = re.sub(r"\bwilln't\b", "won't", string, flags=re.IGNORECASE)
    string = re.sub(r"\bbeen been\b", "been", string, flags=re.IGNORECASE)
    return string


def allowed_features(spec: dict[str, Any], dialect: Any) -> list[str]:
    """Features this group may use, in Multi-VALUE's application order.

    For a named variety: features with eWAVE attestation >= min_attestation, minus `exclude`.
    For a feature list (L2 English): exactly the listed features, minus `exclude`.
    """
    exclude = set(spec.get("exclude", []))
    if spec["kind"] == "multivalue":
        min_att = spec.get("min_attestation", 1.0)
        feats = [f for f, att in dialect.morphosyntax_transforms.items() if att >= min_att]
    else:
        feats = list(spec["features"])
    return [f for f in feats if f not in exclude]


class MultiValueTransformer:
    """Wraps a Multi-VALUE dialect, either a named eWAVE variety or an explicit feature list.

    spec examples (see configs/data.yaml):
      {"kind": "multivalue", "dialect": "AfricanAmericanVernacular", "min_attestation": 1.0,
       "exclude": ["yall"], "feature_prob": 0.5}
      {"kind": "multivalue_features", "features": ["remove_det_definite", "zero_plural"], "feature_prob": 0.5}
    """

    def __init__(self, name: str, spec: dict[str, Any]):
        _skip_spacy_download_if_installed()
        from multivalue import Dialects  # heavy import: spaCy + stanza pipelines

        self.name = name
        self.spec = spec
        if spec["kind"] == "multivalue":
            self._dialect = getattr(Dialects, spec["dialect"])()
        elif spec["kind"] == "multivalue_features":
            self._dialect = Dialects.DialectFromFeatureList(feature_list=spec["features"], dialect_name=name)
        else:
            raise ValueError(f"Unknown dialect spec kind: {spec['kind']}")
        self.allowed = allowed_features(spec, self._dialect)
        if not self.allowed:
            raise ValueError(f"No features left for {name} after filtering")
        self.feature_prob = float(spec.get("feature_prob", 1.0))
        # Replace the slang-adding contraction pass with its repair-only half.
        self._dialect.surface_subs = [
            _repair_only_contractions if getattr(f, "__name__", "") == "surface_contract" else f
            for f in self._dialect.surface_subs
        ]

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]:
        rng = random.Random(seed)
        active = [f for f in self.allowed if rng.random() < self.feature_prob] or [rng.choice(self.allowed)]
        self._dialect.morphosyntax_transforms = {f: 1.0 for f in active}
        # Some features choose between forms with the global `random` module; seed it per message so
        # every variant is reproducible.
        random.seed(seed)
        out = self._dialect.transform(text)
        counter = self._dialect.modification_counter
        features = {k: int(v) for k, v in counter.items() if v and k != "total"}
        return out, features


def make_transformer(name: str, spec: dict[str, Any]) -> Transformer:
    if spec["kind"] == "identity":
        return IdentityTransformer(name)
    return MultiValueTransformer(name, spec)
