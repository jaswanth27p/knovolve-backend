import json
from unittest.mock import patch, MagicMock
from botocore.exceptions import ClientError
from app.storage import s3


def _not_found_error():
    return ClientError({"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}, "HeadBucket")


def test_ensure_bucket_creates_when_missing_and_applies_public_read_policy():
    mock_client = MagicMock()
    mock_client.head_bucket.side_effect = _not_found_error()

    with patch("app.storage.s3._client", return_value=mock_client):
        s3.ensure_bucket()

    mock_client.create_bucket.assert_called_once()
    policy = json.loads(mock_client.put_bucket_policy.call_args.kwargs["Policy"])
    assert policy["Statement"][0]["Resource"] == ["arn:aws:s3:::knovolve/images/*"]


def test_ensure_bucket_skips_create_when_already_exists():
    mock_client = MagicMock()  # head_bucket succeeds (no exception) -> bucket exists

    with patch("app.storage.s3._client", return_value=mock_client):
        s3.ensure_bucket()

    mock_client.create_bucket.assert_not_called()
    mock_client.put_bucket_policy.assert_called_once()


def test_upload_object_calls_put_object():
    mock_client = MagicMock()
    with patch("app.storage.s3._client", return_value=mock_client):
        s3.upload_object("images/x.svg", b"<svg/>", "image/svg+xml")

    mock_client.put_object.assert_called_once_with(
        Bucket="knovolve", Key="images/x.svg", Body=b"<svg/>", ContentType="image/svg+xml"
    )


def test_get_public_url_uses_public_endpoint_and_bucket():
    assert s3.get_public_url("images/x.svg") == "http://localhost:9000/knovolve/images/x.svg"
