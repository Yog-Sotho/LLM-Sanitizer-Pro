"""Structured logging helpers."""
import json
import logging
from datetime import datetime, timezone


class JsonLogFormatter(logging.Formatter):
    """Render log records as one JSON object per line (JSON Lines).

    Fields: ts (ISO-8601 UTC), level, logger, msg, plus exc_info when present.
    Suitable for ingestion by log pipelines running the sanitizer as a service.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            'ts': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
        }
        if record.exc_info:
            payload['exc'] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)
