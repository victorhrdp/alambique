import os
import json
import logging
from pathlib import Path
from .base import BaseTranscriptProvider

logger = logging.getLogger("alambique.transcripts.antigravity_cli")

class AntigravityCliProvider(BaseTranscriptProvider):
    def can_handle(self, conversation_id: str | None = None, client: str | None = None) -> bool:
        if client and client != "antigravity_cli":
            return False
        conv_id = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        if not conv_id:
            return False
        log_dir = Path.home() / ".gemini" / "antigravity-cli" / "brain" / conv_id
        return log_dir.exists()

    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        conv_id = conversation_id or os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
        if not conv_id:
            logger.warning("AntigravityCliProvider: No conversation ID provided or found in environment.")
            return []

        log_dir = Path.home() / ".gemini" / "antigravity-cli" / "brain" / conv_id / ".system_generated" / "logs"
        full_path = log_dir / "transcript_full.jsonl"
        normal_path = log_dir / "transcript.jsonl"

        path = full_path if full_path.exists() else normal_path
        if not path.exists():
            logger.warning("AntigravityCliProvider: Transcript file not found in %s", log_dir)
            return []

        messages = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    step = json.loads(line)
                except Exception:
                    continue
                step_type = step.get("type")
                source = step.get("source")
                content = step.get("content", "")

                # Mensaje del usuario
                if step_type == "USER_INPUT" or source == "USER_EXPLICIT":
                    clean_content = content
                    if "<USER_REQUEST>" in content and "</USER_REQUEST>" in content:
                        try:
                            start = content.find("<USER_REQUEST>") + len("<USER_REQUEST>")
                            end = content.find("</USER_REQUEST>")
                            clean_content = content[start:end].strip()
                        except Exception:
                            pass
                    if clean_content:
                        messages.append({"role": "user", "content": clean_content})

                # Mensaje del asistente (respuesta final al usuario)
                elif step_type == "PLANNER_RESPONSE" or source == "MODEL":
                    if not step.get("tool_calls") and content:
                        messages.append({"role": "assistant", "content": content})

        return messages
