from app.config import Settings


def test_langfuse_settings_default_off():
    s = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]
    assert s.langfuse_enabled is False
    assert s.langfuse_host == "http://localhost:3400"
    assert s.langfuse_tracing_environment == "development"
