import time
import hashlib
import zenoh
import os
from pathlib import Path

CHUNK_SIZE = 20 * 1024  # 128KB chunks
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, "..", "test_image", "crying_cat.jpg")


def query_handler(query: zenoh.Query):
    print(SCRIPT_DIR)
    print(f">> [Queryable ] Received Query '{query.selector}'")
    if not os.path.exists(file_path):
        print(f"No such file exists on {file_path}")
        query.reply(query.key_expr, b"", encoding=zenoh.Encoding.TEXT_JSON5)
        return

    file_size = os.path.getsize(file_path)
    sha256_hash = hashlib.sha256()
    file_data = None
    with open(file_path, "rb") as f:
        file_data = f.read()
        sha256_hash.update(file_data)

    # while chunk := f.read(CHUNK_SIZE):
    file_hash = sha256_hash.hexdigest()
    print(f"SENT file_hash {file_hash}")

    # metadata = f"{file_hash}|{file_size}".encode()
    # query.reply(query.key_expr, metadata, encoding=zenoh.Encoding.TEXT_JSON5)
    # print(f"SENT metadata {metadata}")

    # Send chunks with index
    start_time = time.time()
    query.reply(query.key_expr, file_data, encoding=zenoh.Encoding.TEXT_JSON5)
    end_time = time.time()
    duration = end_time - start_time
    print(
        f">> [Queryable] Transfer complete: {duration:.5f}s, {(file_size / duration) / (1024*1024):.2f} MB/s"
    )


def main(conf: zenoh.Config, key: str):
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    print("Opening session...")
    with zenoh.open(conf) as session:
        print(f"File Service active on '{key}'...")
        session.declare_queryable(key, query_handler)

        print("Press CTRL-C to quit...")
        while True:
            time.sleep(1)


# --- Command line argument parsing --- --- --- --- --- ---
if __name__ == "__main__":
    import argparse
    import os

    import common

    parser = argparse.ArgumentParser(
        prog="z_big_file_queryable", description="BIG FILE"
    )
    common.add_config_arguments(parser)
    parser.add_argument(
        "--key",
        "-k",
        dest="key",
        default="BIG/**",
        type=str,
        help="The key expression to subscribe to.",
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(conf, args.key)
