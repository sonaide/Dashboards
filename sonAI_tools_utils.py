import os
import boto3
import warnings

def _extract_microphone_from_filename(audio_filename):
    """Extraction the microphone/device name from an audio filename

    :param str audio_filename: audio filename (only the filename, NO PATH)
    :return str: microphone/device type, e.g., samsung-tablet, pebble, etc.
    """
    if audio_filename.startswith("ID-"):
        microphone = "-".join(audio_filename.split("_")[2].split("-")[1:])
    elif audio_filename.startswith("PersID-"):
        microphone = "-".join(audio_filename.split("_")[1].split("-")[1:])
        if microphone != "samsung":
            # TODO when we can match a tablet ID with a tablet type
            microphone = "samsung-tablet"
    else:
        raise NotImplementedError("Audio filename format not recognized.")

    return microphone

def _extract_person_from_filename(audio_filename):
    """Extract the person name or ID from an audio filename

    :param str audio_filename: audio filename (only the filename, NO PATH)
    :return str: person name (e.g., "PA") or ID (e.g., "406")
    """
    person = audio_filename.split("_")[0].split("-")[1]
    return person


def download_from_s3(
    bucket_name,
    source_filepath,
    target_filepath,
    endpoint_url="https://s3.fr-par.scw.cloud",
):
    """
    Download a file from an S3 bucket.

    :param str bucket_name: Name of the S3 bucket
    :param str | LiteralString | bytes source_filepath: Key of the file to download from S3
    :param str target_filepath: Local path to save the downloaded file
    :param str endpoint_url: Endpoint URL of the S3 service (default is for Scaleway S3)
    """
    if os.path.exists(target_filepath):
        warnings.warn(f"File already exists : {target_filepath}")
        return
    if (
        not os.path.exists(os.path.dirname(target_filepath))
        and os.path.dirname(target_filepath) != ""
    ):
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    # if not os.path.exists(target_filepath + source_filepath.split('/')[-1]):
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    s3.download_file(bucket_name, source_filepath, target_filepath)
    print(f"Downloaded {source_filepath} to {target_filepath}")

def list_files_in_s3_folder(
    bucket_name,
    prefix="",
    delimiter="",
    filter_file_type="",
    endpoint_url="https://s3.fr-par.scw.cloud",
):
    """
    List all files in a specific folder in an S3 bucket.

    :param str bucket_name: Name of the S3 bucket
    :param str folder_name: Key of the folder to list from S3
    :param str endpoint_url: Endpoint URL of the S3 service (default is for Scaleway S3)
    :param str prefix: Prefix for file name
    :return list: List of file keys
    """
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    results = []
    paginator = s3.get_paginator("list_objects_v2")
    if prefix != "" and not prefix.endswith("/"):
        prefix = prefix + "/"

    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(filter_file_type):
                results.append(key)
    return results