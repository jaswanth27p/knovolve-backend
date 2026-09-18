from langchain_core.tools import tool
from pydantic import BaseModel

from app.config import settings
from app.llm.factory import get_chat_model


@tool
def _dummy_tool(x: str) -> str:
    """A dummy tool used only to exercise .bind_tools()."""
    return x


class _DummySchema(BaseModel):
    answer: str


def test_get_chat_model_binds_langfuse_handler_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    import app.llm.langfuse_client as lc
    monkeypatch.setattr(lc, "_handler_built", False)
    monkeypatch.setattr(lc, "_handler", None)

    model = get_chat_model("chat_reply")
    # callbacks/metadata are bound as native Pydantic fields directly on the
    # ChatOpenAI instance (NOT via .with_config(), which wraps the model in a
    # RunnableBinding — see the comment in app/llm/factory.py for why that
    # breaks bind_tools()/with_structured_output()). get_chat_model's
    # declared return type is plain ChatOpenAI, so these attributes are
    # always present per that type; asserting on them directly (no getattr
    # fallback) is intentional and matches the real return type now.
    assert model.callbacks, "expected a callback handler bound when langfuse_enabled=True"
    assert model.metadata == {"langfuse_tags": ["chat_reply"]}


def test_get_chat_model_no_callbacks_when_disabled(monkeypatch):
    monkeypatch.setattr(settings, "langfuse_enabled", False)

    model = get_chat_model("chat_reply")
    assert not model.callbacks


def test_bind_tools_preserves_langfuse_callback(monkeypatch):
    """Regression test: model.with_config({"callbacks": [handler]}) followed
    by .bind_tools(...) silently drops the callback, because
    RunnableBinding.__getattr__ only merges its own .config into delegated
    methods whose signature accepts a `config` kwarg — bind_tools() doesn't
    have one, so it's forwarded straight to the *unwrapped* inner model, and
    that model's own `self.bind(...)` call (inside bind_tools) builds a
    fresh RunnableBinding with an empty .config. get_chat_model() avoids this
    entirely by setting callbacks/metadata as native fields on the ChatOpenAI
    instance itself, so every RunnableBinding bind_tools()/
    with_structured_output() construct on top of it still wraps a model that
    already carries them. This test would have failed under the old
    .with_config()-based implementation."""
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    import app.llm.langfuse_client as lc
    monkeypatch.setattr(lc, "_handler_built", False)
    monkeypatch.setattr(lc, "_handler", None)

    model = get_chat_model("chat_reply")
    bound = model.bind_tools([_dummy_tool])
    # bound is a RunnableBinding wrapping the same (already-configured)
    # ChatOpenAI instance, so the callback must still be reachable via the
    # underlying bound model, not bound.config (which bind_tools leaves
    # empty regardless of this fix — the fix works by not depending on it).
    # bind_tools()'s declared return type is the generic
    # Runnable[LanguageModelInput, AIMessage], which has no `.bound` member —
    # true at runtime only for a RunnableBinding, which this always is here.
    assert bound.bound.callbacks, "bind_tools() must not drop the langfuse callback"  # pyright: ignore[reportAttributeAccessIssue]


def test_with_structured_output_preserves_langfuse_callback(monkeypatch):
    """Same regression as test_bind_tools_preserves_langfuse_callback, for
    with_structured_output() — its own source is also a plain
    `return self.bind(...)`/`self.bind(**bind_kwargs)` call with no `config`
    parameter, so it has the identical drop-the-callback failure mode."""
    monkeypatch.setattr(settings, "langfuse_enabled", True)
    monkeypatch.setattr(settings, "langfuse_public_key", "pk-lf-test")
    monkeypatch.setattr(settings, "langfuse_secret_key", "sk-lf-test")
    import app.llm.langfuse_client as lc
    monkeypatch.setattr(lc, "_handler_built", False)
    monkeypatch.setattr(lc, "_handler", None)

    model = get_chat_model("chat_reply")
    structured = model.with_structured_output(_DummySchema)
    # with_structured_output returns `llm | output_parser` (a
    # RunnableSequence); the first step is the bind_tools-style binding over
    # the same underlying model. Declared return type is the generic
    # Runnable[LanguageModelInput, _DictOrPydantic], which has no `.steps`
    # member — true at runtime only for a RunnableSequence, which this
    # always is for the (default) non-include_raw path used here.
    llm_step = structured.steps[0]  # pyright: ignore[reportAttributeAccessIssue]
    assert llm_step.bound.callbacks, "with_structured_output() must not drop the langfuse callback"
