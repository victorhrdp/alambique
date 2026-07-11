import abc

class BaseTranscriptProvider(abc.ABC):
    @abc.abstractmethod
    def can_handle(self, conversation_id: str | None = None, client: str | None = None) -> bool:
        """Retorna True si este proveedor puede operar en el entorno actual o con el ID dado."""
        pass

    @abc.abstractmethod
    def get_messages(self, conversation_id: str | None = None) -> list[dict[str, str]]:
        """Extrae y devuelve los mensajes de la conversación actual."""
        pass
