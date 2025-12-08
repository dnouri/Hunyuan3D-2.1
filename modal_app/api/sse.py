"""SSE (Server-Sent Events) formatting and parsing.

Server-side (formatting):
    from modal_app.api.sse import sse_event, sse_progress, sse_error

    yield sse_started("job-123")
    yield sse_progress("shape", 50)
    yield sse_completed("job-123", "https://...")

Client-side (parsing):
    from modal_app.api.sse import parse_sse_line

    for line in response.iter_lines():
        line_type, value = parse_sse_line(line)
        if line_type == "event":
            current_event = value
        elif line_type == "data":
            handle_event(current_event, value)
"""

from __future__ import annotations

import json


def sse_event(event_type: str, data: dict) -> str:
    """Format a Server-Sent Event message.

    Args:
        event_type: Event type (e.g., "started", "progress", "completed", "error")
        data: Dictionary to serialize as JSON in the data field

    Returns:
        SSE-formatted string ready to yield from StreamingResponse
    """
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def sse_started(job_id: str) -> str:
    """Format a 'started' event indicating job has begun.

    Args:
        job_id: Unique identifier for this generation job

    Returns:
        SSE-formatted started event
    """
    return sse_event("started", {"job_id": job_id})


def sse_progress(stage: str, percent: int) -> str:
    """Format a 'progress' event with stage and completion percentage.

    Args:
        stage: Current processing stage (e.g., "shape", "texture", "uploading")
        percent: Completion percentage (0-100)

    Returns:
        SSE-formatted progress event
    """
    return sse_event("progress", {"stage": stage, "percent": percent})


def sse_completed(job_id: str, download_url: str) -> str:
    """Format a 'completed' event with download URL.

    Args:
        job_id: Unique identifier for the completed job
        download_url: Presigned S3 URL for downloading the result

    Returns:
        SSE-formatted completed event
    """
    return sse_event("completed", {"job_id": job_id, "download_url": download_url})


def sse_error(stage: str, message: str, *, retriable: bool = False) -> str:
    """Format an 'error' event with details and retriability.

    Args:
        stage: Stage where error occurred
        message: Human-readable error message
        retriable: Whether client should retry (default: False, fail fast)

    Returns:
        SSE-formatted error event
    """
    return sse_event(
        "error",
        {
            "stage": stage,
            "message": message,
            "retriable": retriable,
        },
    )


# =============================================================================
# Client-side parsing
# =============================================================================


def parse_sse_line(line: str) -> tuple[str | None, str | dict | None]:
    """Parse a single SSE line into its type and value.

    This is the inverse of sse_event() - use it to parse SSE streams on the client.

    Args:
        line: A single line from an SSE stream

    Returns:
        Tuple of (line_type, value) where:
        - ("event", event_name) for "event: xxx" lines
        - ("data", parsed_dict) for "data: {json}" lines
        - (None, None) for empty lines, comments, or unrecognized formats
    """
    if line.startswith("event: "):
        return ("event", line[7:])
    elif line.startswith("data: "):
        try:
            return ("data", json.loads(line[6:]))
        except json.JSONDecodeError:
            return ("data", {"raw": line[6:]})
    return (None, None)
