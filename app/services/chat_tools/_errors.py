class ChatToolError(Exception):
    """Raised by a chat tool when it can't fulfill a request — course not
    found/not started, record not owned by this learner, content not ready.
    The agent loop (app.services.chat) catches this and turns it into a
    ToolMessage the model can react to, instead of the request crashing."""
