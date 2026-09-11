class ExportToolError(Exception):
    """Model-facing export-tool failure: missing course, missing enrollment,
    inaccessible chapter/assignment, or content that is not ready. The custom
    export agent converts this into a recoverable tool message."""
