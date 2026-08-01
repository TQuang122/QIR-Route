import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qir_route.baseline import minmax_rowwise, prefer_unique_sample
from qir_route.metrics import evaluate_single_positive

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIRE_SRC = REPOSITORY_ROOT / "upstream" / "ViRE" / "src"
pytestmark = pytest.mark.skipif(
    not VIRE_SRC.is_dir(), reason="pinned ViRE checkout absent"
)


def test_sampling_matches_pinned_vire_for_unique_contexts() -> None:
    sys.path.insert(0, str(VIRE_SRC))
    try:
        from vi_retrieval_eval.sampling import prefer_unique_context_sampling

        frame = pd.read_csv(
            REPOSITORY_ROOT / "upstream" / "ViRE" / "data" / "CSConDa.csv"
        )
        expected = prefer_unique_context_sampling(frame, 64, seed=42)
        observed = prefer_unique_sample(frame, 64, seed=42)
        assert (
            observed["qid"].astype(str).tolist() == expected["qid"].astype(str).tolist()
        )
    finally:
        sys.path.remove(str(VIRE_SRC))


def test_fusion_and_metrics_match_pinned_vire() -> None:
    sys.path.insert(0, str(VIRE_SRC))
    try:
        from vi_retrieval_eval.fusion import minmax_rowwise as vire_minmax
        from vi_retrieval_eval.metrics import evaluate_all

        scores = np.asarray(
            [[0.2, 0.5, 0.1, 0.0], [0.1, 0.4, 0.3, 0.2]], dtype=np.float32
        )
        gold = [1, 2]
        assert np.allclose(minmax_rowwise(scores), vire_minmax(scores))
        ours, _ = evaluate_single_positive(scores, gold, [1, 3])
        theirs = evaluate_all(scores, [[index] for index in gold], ks=[1, 3])
        for metric in ["Recall@1", "MRR@1", "nDCG@1", "Recall@3", "MRR@3", "nDCG@3"]:
            assert np.isclose(ours[metric], theirs[metric])
    finally:
        sys.path.remove(str(VIRE_SRC))
