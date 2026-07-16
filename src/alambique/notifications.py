"""Desktop notifications for daemon runtime events."""

from __future__ import annotations

import logging
import shutil
import subprocess

logger = logging.getLogger("alambique.notifications")

NOTIFY_APP_NAME = "Alambique"
PASS_WAIT_TITLE = "Alambique necesita tu contraseña GPG"
PASS_WAIT_BODY = (
    "Introduce la contraseña en pinentry para desbloquear pass. "
    "Reintentará automáticamente cada 30 segundos."
)


def send_desktop_notification(
    title: str,
    body: str,
    *,
    urgency: str = "normal",
    icon: str = "dialog-password",
) -> bool:
    if not shutil.which("notify-send"):
        logger.debug("notify-send no disponible; omitiendo notificación")
        return False
    try:
        subprocess.run(
            [
                "notify-send",
                "-a",
                NOTIFY_APP_NAME,
                "-i",
                icon,
                f"--urgency={urgency}",
                title,
                body,
            ],
            check=False,
            timeout=5,
        )
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("No se pudo enviar notificación de escritorio: %s", exc)
        return False


def is_waiting_pass_error(error: str | None) -> bool:
    if not error:
        return False
    lowered = error.lower()
    return "pass no respondió" in lowered or "pinentry" in lowered