import logging
from collections import Counter
from datetime import timedelta
from pathlib import Path

import numpy as np
import openai
import pytest
import yaml

from news.digest.llm_client import LlmClient
from news.digest.map_reduce import MapReduceGrouping
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
    sorted_entries = MapReduceGrouping.sort_by_published_at(frozen_entries)
    baseline_cut, clustering_cut = MapReduceGrouping.split_offsets(
        sorted_entries,
        timedelta(hours=eval_settings.grouping_window_hours),
    )

    # Act
    try:
        embedded = await grouping.embed_entries(sorted_entries)
    finally:
        await llm.aclose()
    vectors_all = np.asarray(
        [entry.vector for entry in embedded], dtype=np.float64
    )
    baseline_vectors = vectors_all[:baseline_cut]
    clustering_entries = sorted_entries[clustering_cut:]
    vectors = vectors_all[clustering_cut:]
    for sigma in [2.5, 3.0, 3.5, 2.0]:
        threshold = MapReduceGrouping.calibrate_threshold(
            vectors, n_pairs=20000, k_sigma=sigma
        )
        labels = MapReduceGrouping.cluster(vectors, threshold)
    scored = MapReduceGrouping.score_clusters(
        clustering_entries, labels, vectors, baseline_vectors
    )
    clusters = [
        {
            **{key: value for key, value in group.items() if key != "records"},
            "entries": [
                {"title": entry.title, "link": entry.link}
                for entry in group["records"]
            ],
        }
        for group in scored
    ]

    output_path = (
        Path(__file__).parent.parent.parent / "tmp"
        / f"cluster_eval_{dataset_id}.yaml"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(clusters, sort_keys=False))
    logger.info(
        "cluster eval written path=%s clusters_count=%s threshold=%.4f",
        output_path,
        len(clusters),
        threshold,
    )

    # Assert
    assert clusters, "clustering produced no groups"
    loaded = yaml.safe_load(output_path.read_text())
    metadata = {
        "size", "unique_domains", "source_entropy", "effective_sources",
        "burst_z", "novelty", "trend_score",
    }
    assert all(metadata <= group.keys() for group in loaded)
    assert all(group["size"] == len(group["entries"]) for group in loaded)
    assert Counter(
        (entry["title"], entry["link"])
        for group in loaded for entry in group["entries"]
    ) == Counter(
        (entry.title, entry.link) for entry in clustering_entries
    )
    scores = [group["trend_score"] for group in loaded]
    assert scores == sorted(scores, reverse=True)
