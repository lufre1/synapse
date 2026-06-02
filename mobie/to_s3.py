import argparse
from mobie.metadata.remote_metadata import add_remote_project_metadata


def parse_args():
    p = argparse.ArgumentParser(
        description="Add/overwrite remote S3 location metadata for a local MoBIE project."
    )
    p.add_argument(
        "--root", required=True,
        help="Local MoBIE project root folder (contains mobie.json and dataset folders)."
    )
    p.add_argument(
        "--bucket", required=True,
        help="S3 bucket name, optionally with prefix (e.g. 'volume-em' or 'volume-em/mitochondria')."
    )
    # old endpoint for GoeNet
    # p.add_argument(
    #     "--endpoint", default="https://s3.fs.gwdg.de",
    #     help="S3 service endpoint URL (default: https://s3.fs.gwdg.de)."
    # )
    # public domain
    p.add_argument(
        "--endpoint", default="https://s3.gwdg.de",
        help="S3 service endpoint URL (default: https://s3.gwdg.de)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    endpoint = args.endpoint.strip()  # avoid trailing spaces
    add_remote_project_metadata(args.root, args.bucket, endpoint)


if __name__ == "__main__":
    main()