"""Tests for human-readable warning labels."""

from alambique.warning_labels import (
    format_warnings_for_humans,
    humanize_warning,
    is_benign_consolidation_only,
    warning_message_level,
)


class TestHumanizeWarning:
    def test_known_health_code(self):
        assert humanize_warning("ollama_unavailable") == "Ollama no responde"

    def test_consolidation_prefix(self):
        msg = humanize_warning("consolidation_filtered_prefix:architecture_")
        assert "arquitectura del código" in msg
        assert "consolidar" in msg

    def test_consolidation_substring(self):
        msg = humanize_warning("consolidation_filtered_substring:_changelog")
        assert "efímera" in msg


class TestBenignConsolidationLevel:
    def test_consolidation_only_is_benign(self):
        codes = ["consolidation_filtered_prefix:architecture_"]
        assert is_benign_consolidation_only(codes) is True

    def test_unanchored_thread_is_benign_filter(self):
        codes = ["consolidation_thread_unanchored:english_lessons_flirting"]
        assert is_benign_consolidation_only(codes) is True
        msg = humanize_warning(codes[0])
        assert "english_lessons_flirting" in msg
        assert "anclado" in msg
        assert warning_message_level(codes) == "info"

    def test_empty_result_is_not_benign(self):
        codes = ["consolidation_empty_result"]
        assert is_benign_consolidation_only(codes) is False
        assert warning_message_level(codes) == "warning"
        assert "vacía" in humanize_warning(codes[0]).lower() or "JSON" in humanize_warning(codes[0])

    def test_mixed_is_not_benign(self):
        codes = ["ollama_unavailable", "consolidation_filtered_prefix:task_"]
        assert is_benign_consolidation_only(codes) is False
        assert warning_message_level(codes) == "warning"


class TestFormatWarningsForHumans:
    def test_single_consolidation_filter(self):
        msg = format_warnings_for_humans(["consolidation_filtered_prefix:architecture_"])
        assert "Todo funciona" in msg
        assert "arquitectura del código" in msg
        assert "consolidation_filtered" not in msg

    def test_multiple_consolidation_filters(self):
        msg = format_warnings_for_humans(
            [
                "consolidation_filtered_prefix:architecture_",
                "consolidation_filtered_prefix:task_",
            ]
        )
        assert "descarté 2 notas" in msg

    def test_mixed_warnings(self):
        msg = format_warnings_for_humans(
            ["ollama_unavailable", "consolidation_filtered_prefix:task_"]
        )
        assert msg.startswith("Funcionando con avisos:")
        assert "Ollama no responde" in msg
        assert "consolidation_filtered" not in msg