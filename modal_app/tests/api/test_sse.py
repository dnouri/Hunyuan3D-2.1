"""Tests for SSE response helpers.

These helpers format Server-Sent Events according to the spec:
    event: {type}\n
    data: {json}\n
    \n
"""

import json


class TestSseEvent:
    """Tests for the sse_event() helper function."""

    def test_formats_event_with_type_and_data(self):
        """Event should have event: and data: lines followed by blank line."""
        from modal_app.api.sse import sse_event

        result = sse_event("started", {"job_id": "abc123"})

        assert result == 'event: started\ndata: {"job_id": "abc123"}\n\n'

    def test_data_is_valid_json(self):
        """The data line should contain valid JSON."""
        from modal_app.api.sse import sse_event

        result = sse_event("progress", {"percent": 50, "stage": "shape"})

        # Extract the data line
        lines = result.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        json_str = data_line[6:]  # Remove "data: " prefix

        # Should parse without error
        parsed = json.loads(json_str)
        assert parsed == {"percent": 50, "stage": "shape"}

    def test_handles_nested_data(self):
        """Should handle nested dictionaries in data."""
        from modal_app.api.sse import sse_event

        data = {"job_id": "abc", "metadata": {"size": 1024, "format": "glb"}}
        result = sse_event("completed", data)

        assert "metadata" in result
        assert '"size": 1024' in result

    def test_ends_with_double_newline(self):
        """SSE events must end with \\n\\n per spec."""
        from modal_app.api.sse import sse_event

        result = sse_event("test", {"key": "value"})

        assert result.endswith("\n\n")


class TestSseProgressEvent:
    """Tests for the sse_progress() convenience helper."""

    def test_creates_progress_event_with_stage_and_percent(self):
        """Should create properly formatted progress event."""
        from modal_app.api.sse import sse_progress

        result = sse_progress("shape", 50)

        assert result.startswith("event: progress\n")
        assert '"stage": "shape"' in result
        assert '"percent": 50' in result

    def test_percent_is_integer(self):
        """Percent should be an integer, not float."""
        from modal_app.api.sse import sse_progress

        result = sse_progress("texture", 75)

        # Parse and check type
        lines = result.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        parsed = json.loads(data_line[6:])

        assert isinstance(parsed["percent"], int)
        assert parsed["percent"] == 75


class TestSseErrorEvent:
    """Tests for the sse_error() convenience helper."""

    def test_creates_error_event_with_required_fields(self):
        """Error event should have stage, message, and retriable."""
        from modal_app.api.sse import sse_error

        result = sse_error("shape", "GPU out of memory")

        assert result.startswith("event: error\n")
        assert '"stage": "shape"' in result
        assert '"message": "GPU out of memory"' in result
        assert '"retriable":' in result

    def test_default_retriable_is_false(self):
        """Errors should be non-retriable by default (fail fast)."""
        from modal_app.api.sse import sse_error

        result = sse_error("upload", "S3 connection failed")

        lines = result.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        parsed = json.loads(data_line[6:])

        assert parsed["retriable"] is False

    def test_can_mark_error_as_retriable(self):
        """Some errors may be retriable (e.g., transient network issues)."""
        from modal_app.api.sse import sse_error

        result = sse_error("upload", "Timeout", retriable=True)

        lines = result.strip().split("\n")
        data_line = next(line for line in lines if line.startswith("data: "))
        parsed = json.loads(data_line[6:])

        assert parsed["retriable"] is True


class TestSseStartedEvent:
    """Tests for the sse_started() convenience helper."""

    def test_creates_started_event_with_job_id(self):
        """Started event should include the job_id."""
        from modal_app.api.sse import sse_started

        result = sse_started("job-12345678")

        assert result.startswith("event: started\n")
        assert '"job_id": "job-12345678"' in result


class TestSseCompletedEvent:
    """Tests for the sse_completed() convenience helper."""

    def test_creates_completed_event_with_job_id_and_url(self):
        """Completed event should include job_id and download_url."""
        from modal_app.api.sse import sse_completed

        url = "https://bucket.s3.amazonaws.com/results/job-123.glb?signature=xxx"
        result = sse_completed("job-123", url)

        assert result.startswith("event: completed\n")
        assert '"job_id": "job-123"' in result
        assert '"download_url":' in result
        assert "bucket.s3.amazonaws.com" in result


class TestParseSseLine:
    """Tests for the parse_sse_line() client-side parser."""

    def test_parses_event_line(self):
        """Should extract event type from event: line."""
        from modal_app.api.sse import parse_sse_line

        line_type, value = parse_sse_line("event: progress")

        assert line_type == "event"
        assert value == "progress"

    def test_parses_data_line_with_valid_json(self):
        """Should parse data: line as JSON."""
        from modal_app.api.sse import parse_sse_line

        line_type, value = parse_sse_line('data: {"stage": "shape", "percent": 50}')

        assert line_type == "data"
        assert value == {"stage": "shape", "percent": 50}

    def test_handles_invalid_json_gracefully(self):
        """Invalid JSON should be returned as raw string in dict."""
        from modal_app.api.sse import parse_sse_line

        line_type, value = parse_sse_line("data: not valid json")

        assert line_type == "data"
        assert value == {"raw": "not valid json"}

    def test_returns_none_for_empty_lines(self):
        """Empty lines should return (None, None)."""
        from modal_app.api.sse import parse_sse_line

        line_type, value = parse_sse_line("")

        assert line_type is None
        assert value is None

    def test_returns_none_for_comments(self):
        """SSE comment lines (: prefix) should return (None, None)."""
        from modal_app.api.sse import parse_sse_line

        line_type, value = parse_sse_line(": this is a comment")

        assert line_type is None
        assert value is None

    def test_roundtrip_with_sse_event(self):
        """Parsing an event formatted by sse_event should recover the data."""
        from modal_app.api.sse import parse_sse_line, sse_event

        original_data = {"job_id": "abc123", "status": "complete"}
        formatted = sse_event("completed", original_data)

        # Parse each line
        lines = formatted.strip().split("\n")
        event_type = None
        parsed_data = None

        for line in lines:
            line_type, value = parse_sse_line(line)
            if line_type == "event":
                event_type = value
            elif line_type == "data":
                parsed_data = value

        assert event_type == "completed"
        assert parsed_data == original_data
