from app.config import settings
from app.llm.factory import get_chat_model


def test_get_chat_model_binds_langfuse_handler_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    import app.llm.langfuse_client as lc
    monkeypatch.setattr(lc, "_handler_built", False)
    monkeypatch.setattr(lc, "_handler", None)

    model = get_chat_model("chat_reply")
    # get_chat_model's return type is ChatOpenAI | Runnable (a RunnableBinding
    # once .with_config(...) wraps it), so use getattr rather than assume
    # `.config` exists on the static union type — it always exists at
    # runtime here since langfuse_enabled=True forces the RunnableBinding
    # branch, which is exactly what this test is asserting.
    config = getattr(model, "config", None)
    bound_callbacks = config.get("callbacks") if config else None
    assert bound_callbacks, "expected a callback handler bound when langfuse_enabled=True"


def test_get_chat_model_no_callbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)

    model = get_chat_model("chat_reply")
    # When disabled, get_chat_model must return the bare ChatOpenAI with no
    # .with_config(...) wrapping (same object identity/behavior as before
    # this change) — a plain ChatOpenAI has no `.config` attribute at all
    # (only a RunnableBinding produced by .with_config does), so this must
    # use getattr rather than assume the attribute exists.
    config = getattr(model, "config", None)
    bound_callbacks = config.get("callbacks") if config else None
    assert not bound_callbacks
