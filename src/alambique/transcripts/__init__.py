from .base import BaseTranscriptProvider
from .antigravity_cli import AntigravityCliProvider
from .grok_cli import GrokCliProvider
from .opencode_cli import OpenCodeCliProvider

PROVIDERS: list[BaseTranscriptProvider] = [
    AntigravityCliProvider(),
    GrokCliProvider(),
    OpenCodeCliProvider(),
]

def get_active_provider(conversation_id: str | None = None, client: str | None = None) -> BaseTranscriptProvider | None:
    for provider in PROVIDERS:
        if provider.can_handle(conversation_id, client):
            return provider
    return None
