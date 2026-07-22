"""Loncha B: consolidation prompt thread selection (not high-salience dump)."""

from alambique.consolidation_retrieval import (
    format_threads_for_prompt,
    merge_thread_candidates,
    select_threads_for_consolidation_prompt,
    threads_from_vector_hits,
)


def _t(key: str, **extra) -> dict:
    row = {"key": key, "title": key, "current_state": "x" * 40, "salience": 0.5}
    row.update(extra)
    return row


class TestMergeCandidates:
    def test_similar_before_recent_and_dedup(self):
        similar = [_t("a"), _t("b")]
        recent = [_t("b"), _t("c"), _t("d")]
        out = merge_thread_candidates(similar=similar, recent=recent, list_cap=10)
        assert [t["key"] for t in out] == ["a", "b", "c", "d"]

    def test_list_cap(self):
        similar = [_t(f"s{i}") for i in range(8)]
        recent = [_t(f"r{i}") for i in range(8)]
        out = merge_thread_candidates(similar=similar, recent=recent, list_cap=5)
        assert len(out) == 5
        assert [t["key"] for t in out] == ["s0", "s1", "s2", "s3", "s4"]

    def test_empty(self):
        assert merge_thread_candidates(similar=[], recent=[]) == []


class TestVectorHits:
    def test_drops_weak_neighbors(self):
        hits = [
            {"thread": _t("close"), "distance": 0.2},
            {"thread": _t("far"), "distance": 9.0},
            {"thread": _t("mid"), "distance": 1.0},
        ]
        out = threads_from_vector_hits(hits, max_distance=1.5, limit=8)
        assert [t["key"] for t in out] == ["close", "mid"]

    def test_limit(self):
        hits = [{"thread": _t(f"k{i}"), "distance": 0.1} for i in range(20)]
        out = threads_from_vector_hits(hits, max_distance=2.0, limit=3)
        assert len(out) == 3


class TestSelectAndFormat:
    def test_select_prefers_similar(self):
        hits = [{"thread": _t("philosophy"), "distance": 0.3}]
        recent = [_t("english_lessons_flirting", salience=0.99), _t("gpu_pcie")]
        out = select_threads_for_consolidation_prompt(
            similar_hits=hits,
            recent_threads=recent,
            list_cap=10,
        )
        assert out[0]["key"] == "philosophy"
        assert "english_lessons_flirting" in [t["key"] for t in out]

    def test_format_empty(self):
        assert format_threads_for_prompt([]) == "(ninguno relevante)"

    def test_format_includes_key(self):
        text = format_threads_for_prompt([_t("alambique_memory_architecture", title="Mem")])
        assert "alambique_memory_architecture" in text
        assert "Mem" in text
