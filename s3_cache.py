import boto3  # type: ignore
import os
import time
from aws_lambda_powertools import Logger
from typing import Dict

logger = Logger()

s3_client = boto3.client("s3")
S3_BUCKET = os.environ.get("S3_BUCKET_NAME", "your-default-bucket-name")
CACHE_PREFIX = "cache/"
MAX_SIZES: Dict[str, int] = {}


def _get_full_key(namespace: str, key: str) -> str:
    return f"{CACHE_PREFIX}{namespace}/{key}"


def get_cache(namespace: str, key: str) -> str | None:
    full_key = _get_full_key(namespace, key)
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=full_key)
        return response["Body"].read().decode("utf-8")
    except s3_client.exceptions.NoSuchKey:
        logger.info(f"Cache miss for key: {full_key}")
        return None
    except Exception as e:
        logger.error(f"Error retrieving cache for key {full_key}: {str(e)}")
        return None


def set_cache(namespace: str, key: str, value: str) -> None:
    full_key = _get_full_key(namespace, key)
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=full_key,
            Body=value.encode("utf-8"),
            Metadata={"timestamp": str(int(time.time()))},
        )
        logger.info(f"Cache set for key: {full_key}")

        # Enforce size limit if set for the namespace
        if namespace in MAX_SIZES:
            enforce_size_limit(namespace)
    except Exception as e:
        logger.error(f"Error setting cache for key {full_key}: {str(e)}")


def delete_cache(namespace: str, key: str) -> None:
    full_key = _get_full_key(namespace, key)
    try:
        s3_client.delete_object(Bucket=S3_BUCKET, Key=full_key)
        logger.info(f"Cache deleted for key: {full_key}")
    except Exception as e:
        logger.error(f"Error deleting cache for key {full_key}: {str(e)}")


def set_max_size(namespace: str, max_size: int) -> None:
    MAX_SIZES[namespace] = max_size
    logger.info(f"Max size for namespace {namespace} set to {max_size}")
    enforce_size_limit(namespace)


def enforce_size_limit(namespace: str) -> None:
    max_size = MAX_SIZES.get(namespace)
    if not max_size:
        return

    try:
        # List all objects in the namespace
        objects = []
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=S3_BUCKET, Prefix=f"{CACHE_PREFIX}{namespace}/"
        ):
            if "Contents" in page:
                objects.extend(page["Contents"])

        # Sort objects by timestamp (oldest first)
        objects.sort(
            key=lambda x: int(
                s3_client.head_object(Bucket=S3_BUCKET, Key=x["Key"])["Metadata"][
                    "timestamp"
                ]
            )
        )

        # Delete oldest objects if the count exceeds the max size
        objects_to_delete = objects[:-max_size] if len(objects) > max_size else []
        for obj in objects_to_delete:
            s3_client.delete_object(Bucket=S3_BUCKET, Key=obj["Key"])
            logger.info(f"Deleted old cache entry: {obj['Key']}")

    except Exception as e:
        logger.error(f"Error enforcing size limit for namespace {namespace}: {str(e)}")
