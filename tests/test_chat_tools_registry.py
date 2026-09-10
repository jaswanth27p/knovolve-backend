from app.db import SessionLocal
from app.services.chat_tools.registry import build_tools


def test_build_tools_returns_all_sixteen_tools_with_unique_names():
    with SessionLocal() as db:
        tools = build_tools(db, user_id=1)
    names = [t.name for t in tools]
    assert len(names) == 16
    assert len(set(names)) == 16
    assert "search_courses" in names
    assert "get_recurring_weak_concepts" in names
    assert "get_user_stats" in names


def test_build_tools_no_arg_tool_is_invocable():
    # Invoke inside the `with` so the session is closed (and its transaction
    # rolled back) before teardown — the tools closure over `db`, so invoking
    # after the block leaves that connection idle-in-transaction and the
    # autouse clean_db TRUNCATE blocks on its lock forever.
    with SessionLocal() as db:
        tools = build_tools(db, user_id=1)
        by_name = {t.name: t for t in tools}
        result = by_name["get_user_stats"].invoke({})
    assert "streak_current" in result
