import tarfile
import os
from boto3 import Session
import sagemaker
from sagemaker.huggingface.model import HuggingFaceModel


DEPLOY_DIR = 'deploy'
MODEL_PATH = 'model/'
MODEL_ARCHIVE = 'model.tar.gz'
MODEL_ARCHIVE_LOCAL_PATH = f'{DEPLOY_DIR}/{MODEL_ARCHIVE}'
S3_BUCKET_NAME = "jenet-model"


def compress_model():
    print(f"Compressing {MODEL_PATH} to {MODEL_ARCHIVE_LOCAL_PATH} ...", end="", flush=True)
    with tarfile.open(MODEL_ARCHIVE_LOCAL_PATH, "w:gz") as tar:
        tar.add(MODEL_PATH, arcname=os.path.basename(MODEL_PATH))
    print(f" done")


def upload_model():
    s3_path = f"s3://{S3_BUCKET_NAME}/{MODEL_ARCHIVE}"

    session = Session(profile_name="josh")
    s3 = session.client("s3")
    print(f"Uploading {MODEL_PATH} to {s3_path} ...", end="", flush=True)
    s3.upload_file(MODEL_ARCHIVE_LOCAL_PATH, S3_BUCKET_NAME, MODEL_ARCHIVE)
    print(" done")

    sess = sagemaker.Session(boto_session=session)
    huggingface_model = HuggingFaceModel(
        model_data=s3_path,
        role="arn:aws:iam::958846517764:role/sagemaker",
        transformers_version="4.26",
        pytorch_version="1.13",
        py_version="py39",
        sagemaker_session=sess
    )

    huggingface_model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )
