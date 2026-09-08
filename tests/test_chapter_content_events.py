import json
from unittest.mock import patch, MagicMock
from app.realtime.chapter_content_events import channel_name, publish_event


def test_channel_name_is_stable_and_scoped_to_content_id():
    assert channel_name(42) == "chapter_content:42"
    assert channel_name(42) == channel_name(42)
    assert channel_name(42) != channel_name(43)


def test_publish_event_sends_json_on_the_right_channel():
    mock_client = MagicMock()
    with patch("app.realtime.chapter_content_events._get_client", return_value=mock_client):
        publish_event(42, {"type": "diagram_ready", "order": 0, "diagram_image_url": "http://x/y.svg"})

    args, _ = mock_client.publish.call_args
    assert args[0] == "chapter_content:42"
    assert json.loads(args[1]) == {"type": "diagram_ready", "order": 0, "diagram_image_url": "http://x/y.svg"}
