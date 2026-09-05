import logging
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import openai
import pytest
import yaml

from news.digest.llm_client import LlmClient
from news.digest.map_reduce import EmbeddedEntry, MapReduceGrouping
from news.digest.schemas import RssEntry
from news.settings import Settings

logger = logging.getLogger(__name__)


@pytest.mark.integration
async def test_cluster_eval_writes_inspectable_yaml(
    eval_settings: Settings,
    frozen_entries: list[RssEntry],
    dataset_id: str,
) -> None:
    # Arrange
    llm = LlmClient(
        openai.AsyncOpenAI(
            api_key=eval_settings.litellm_api_key.get_secret_value(),
            base_url=str(eval_settings.litellm_router),
        )
    )
    grouping = MapReduceGrouping(eval_settings, llm)

    # Act
    try:
        embedded = await grouping.embed_entries(frozen_entries)
    finally:
        await llm.aclose()
    title_vectors = np.asarray(
        [entry.title_vector for entry in embedded], dtype=np.float64
    )
    threshold = MapReduceGrouping.calibrate_threshold(title_vectors)
    labels = MapReduceGrouping.cluster(title_vectors, threshold)
    clusters = _clusters_by_size(embedded, labels)

    output_path = Path(gettempdir()) / f"cluster_eval_{dataset_id}.yaml"
    output_path.write_text(yaml.safe_dump(clusters, sort_keys=False))
    logger.info(
        "cluster eval written path=%s clusters_count=%s threshold=%.4f",
        output_path,
        len(clusters),
        threshold,
    )

    # Assert
    assert clusters, "clustering produced no groups"


def _clusters_by_size(
    embedded: list[EmbeddedEntry], labels: np.ndarray
) -> list[dict[str, list[dict[str, str]]]]:
    groups: dict[int, list[dict[str, str]]] = {}
    for entry, label in zip(embedded, labels, strict=True):
        groups.setdefault(int(label), []).append(
            {"title": entry.entry.title, "link": entry.entry.link}
        )
    ordered = sorted(groups.values(), key=len, reverse=True)
    return [{"entries": entries} for entries in ordered]
