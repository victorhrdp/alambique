"""Lexical thread-anchor gate for consolidation apply."""

from alambique.thread_anchor import (
    is_thread_anchored_to_session,
    key_tokens,
    thread_anchor_tokens,
    tokenize,
)


class TestTokenize:
    def test_drops_short_and_generic(self):
        toks = tokenize("Lucy y Victor hablan de inglés y robot")
        assert "lucy" not in toks
        assert "victor" not in toks
        assert "ingles" in toks or "inglés" in toks or "ingles" in {t for t in toks}
        # folded: inglés → ingles
        assert "ingles" in toks
        assert "robot" in toks

    def test_key_tokens_split(self):
        assert "english" in key_tokens("english_lessons_flirting")
        assert "lessons" in key_tokens("english_lessons_flirting")
        assert "familia" in key_tokens("lucy_familia_integration")


class TestAnchorGate:
    def test_english_session_anchors_english_thread(self):
        session = """
        Buff. Hoy tengo el cerebro frito para inglés.
        Read in english is totally ok. Teacher outfit. Vocabulary: cheeky, barely.
        """
        assert is_thread_anchored_to_session(
            session,
            "english_lessons_flirting",
            title="Clases de inglés con flirteo",
            search_text="english lessons flirting vocabulary teacher",
            min_hits=2,
        )

    def test_pinokio_session_rejects_familia_thread(self):
        session = """
        Me gustaría instalar pinokio. Estoy instalando fooocus.
        Cómo funciona la generación de imágenes locales.
        """
        assert not is_thread_anchored_to_session(
            session,
            "lucy_familia_integration",
            title="Integración familiar de Lucy",
            search_text="Noemí hijos familia casa compartida",
            current_state="Noemí conoce a Lucy. Aarón e Isaac preguntan por ella.",
            min_hits=2,
        )

    def test_git_session_rejects_english_thread(self):
        session = """
        Vamos a solucionar lo de git. Haz el push.
        Commit del MVP de iniciativas y fix Unicode.
        """
        assert not is_thread_anchored_to_session(
            session,
            "english_lessons_flirting",
            title="Clases de inglés con flirteo",
            search_text="english lessons flirting melonhead teacher outfit",
            min_hits=2,
        )

    def test_multi_topic_session_anchors_gpu(self):
        session = """
        El PC ya respira. PCIe x8 no x16. nvidia-smi muestra link width.
        También hablamos de consolidación alambique.
        """
        assert is_thread_anchored_to_session(
            session,
            "gpu_pcie_x8_link_issue",
            title="GPU PCIe x8",
            search_text="pcie x8 link width nvidia gpu",
            min_hits=2,
        )

    def test_empty_session_fails_closed(self):
        assert not is_thread_anchored_to_session(
            "",
            "alambique_memory_architecture",
            title="Arquitectura de memoria",
            search_text="consolidación threads embeddings",
        )

    def test_thread_anchor_tokens_uses_key(self):
        anchors = thread_anchor_tokens(
            "philosophy_embodiment_robots",
            title="Encarnación",
            search_text="robot spark cuerpo",
        )
        assert "embodiment" in anchors or "robots" in anchors
        assert "robot" in anchors or "spark" in anchors or "cuerpo" in anchors
