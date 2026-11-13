"""Rule-based dialect variants via Multi-VALUE (Ziems et al., ACL 2023).

Why rule-based: asking an LLM to rewrite text "in AAVE" tends to produce caricature, which would test
the judge's reaction to a stereotype rather than to the dialect. Multi-VALUE applies documented
grammatical features (from the eWAVE atlas), and records which ones fired, so every variant can be
traced feature by feature and each feature can be ablated on its own in stage 3 (E3).

Each transformer returns (new_text, features_applied). Multi-VALUE is imported lazily because it
needs its own environment (pinned spaCy/stanza/nltk); see docs/design_decisions.md.
"""

from __future__ import annotations

import random
from typing import Any, Protocol


class Transformer(Protocol):
    name: str

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]: ...


class IdentityTransformer:
    """SAE: no change. Present so every group goes through the same code path."""

    def __init__(self, name: str = "sae"):
        self.name = name

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]:
        return text, {}


def _skip_spacy_download_if_installed() -> None:
    """Multi-VALUE's __init__ calls spacy.cli.download("en_core_web_sm") on every import, which shells
    out to pip (missing in uv venvs) and races when several shards start at once. setup_local.sh already
    installs the model, so make that call a no-op when the model is present."""
    import spacy
    import spacy.cli

    if spacy.util.is_package("en_core_web_sm"):
        spacy.cli.download = lambda *args, **kwargs: None


class MultiValueTransformer:
    """Wraps a Multi-VALUE dialect, either a named eWAVE variety or an explicit feature list.

    spec examples:
      {"kind": "multivalue", "dialect": "AfricanAmericanVernacular"}
      {"kind": "multivalue_features", "features": ["remove_det_definite", "zero_plural"]}
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

    def transform(self, text: str, seed: int) -> tuple[str, dict[str, int]]:
        # Multi-VALUE applies many features with probability p using the global `random` module,
        # so we seed it per message to make every variant reproducible.
        random.seed(seed)
        out = self._dialect.transform(text)
        counter = self._dialect.modification_counter
        features = {k: int(v) for k, v in counter.items() if v and k != "total"}
        return out, features


def make_transformer(name: str, spec: dict[str, Any]) -> Transformer:
    if spec["kind"] == "identity":
        return IdentityTransformer(name)
    return MultiValueTransformer(name, spec)
