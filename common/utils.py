import os

import sagemaker
from sagemaker import LocalSession, Session


def get_role() -> str:
    role = os.getenv("ROLE")
    if role is None:
        role = sagemaker.get_execution_role()

    return role


def get_training_instance_type() -> str:
    return os.getenv("TRAINING_INSTANCE_TYPE")


def get_processing_instance_type() -> str:
    return os.getenv("PROCESSING_INSTANCE_TYPE")


def get_s3_bucket():
    return os.getenv("S3_BUCKET")


def generate_s3_bucket_prefix_for_job() -> str:
    s3_prefix = os.getenv('S3_PREFIX')
    return f'{s3_prefix}/jobs'


def generate_sagemaker_session(instance_type: str):
    s3_bucket = get_s3_bucket()
    s3_prefix = generate_s3_bucket_prefix_for_job()

    if instance_type in ("local", "local_gpu"):
        return LocalSession(default_bucket=s3_bucket, default_bucket_prefix=s3_prefix)
    else:
        return Session(default_bucket=s3_bucket, default_bucket_prefix=s3_prefix)
