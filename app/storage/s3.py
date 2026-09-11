import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from app.config import settings

#: Only diagram images live under this prefix, and only this prefix is
#: publicly readable (everything else in the bucket stays private) — mirrors
#: the `infinite-draw-backend` project's storage policy pattern.
PUBLIC_READ_PREFIX = "images/"


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint,
        aws_access_key_id=settings.s3_access_key,
        aws_secret_access_key=settings.s3_secret_key,
        config=BotoConfig(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket() -> None:
    client = _client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket)
    except ClientError as exc:
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status != 404:
            raise
        client.create_bucket(Bucket=settings.s3_bucket)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{settings.s3_bucket}/{PUBLIC_READ_PREFIX}*"],
            }
        ],
    }
    import json
    client.put_bucket_policy(Bucket=settings.s3_bucket, Policy=json.dumps(policy))


def upload_object(key: str, data: bytes, content_type: str) -> None:
    _client().put_object(Bucket=settings.s3_bucket, Key=key, Body=data, ContentType=content_type)


def get_public_url(key: str) -> str:
    return f"{settings.s3_public_endpoint}/{settings.s3_bucket}/{key}"


def presign_get_url(key: str, expires: int = 300, download_filename: str | None = None) -> str:
    params = {"Bucket": settings.s3_bucket, "Key": key}
    if download_filename:
        # Force a save-as download with a friendly name. Without this the
        # browser renders the PDF inline (navigation-based downloads) or names
        # the file after the opaque S3 key.
        params["ResponseContentDisposition"] = f'attachment; filename="{download_filename}"'
    return _client().generate_presigned_url(
        "get_object",
        Params=params,
        ExpiresIn=expires,
    )
