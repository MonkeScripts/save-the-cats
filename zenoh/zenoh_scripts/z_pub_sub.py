import time
import argparse
import itertools
import zenoh
from typing import Optional


def main(
    conf: zenoh.Config,
    repub_key: str,
    sub_key: str,
    add_matching_listener: bool,
):
    # initiate logging
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    print("Opening session...")
    with zenoh.open(conf) as session:

        # 1. Declare the Publisher first
        print(f"Declaring Publisher on '{repub_key}'...")
        pub = session.declare_publisher(repub_key)
        if add_matching_listener:

            def on_matching_status_update(status: zenoh.MatchingStatus):
                if status.matching:
                    print("Publisher has matching subscribers.")
                else:
                    print("Publisher has NO MORE matching subscribers")

            pub.declare_matching_listener(on_matching_status_update)

        # 2. Define the callback that does the "re-publishing"
        def republish_callback(sample: zenoh.Sample):
            original_data = sample.payload.to_string()
            new_payload = f"TESTING REPUB: {original_data}"

            print(f">> [Relay] {sample.key_expr} -> {repub_key}: {new_payload}")

            # Re-publish the data to the new key
            pub.put(new_payload)

        # 3. Declare the Subscriber with the relay callback
        print(f"Declaring Subscriber on '{sub_key}'...")
        sub = session.declare_subscriber(sub_key, republish_callback)

        print("Republisher is running. Press CTRL-C to quit...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    import argparse
    import itertools
    import os
    import sys

    _this_dir = os.path.dirname(__file__)
    _repo_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import examples.common as common

    parser = argparse.ArgumentParser(
        prog="z_pub_sub", description="zenoh pub_sub example"
    )
    common.add_config_arguments(parser)

    parser.add_argument(
        "--key",
        "-k",
        dest="repub_key",
        default="demo/repub/zenoh-python-repub",
        type=str,
        help="The key expression to publish onto.",
    )
    parser.add_argument(
        "--sub-key",
        "-s",
        dest="sub_key",
        default="demo/example/**",
        help="Key expression to subscribe to",
    )
    parser.add_argument(
        "--add-matching-listener",
        default=False,
        action="store_true",
        help="Add matching listener",
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(
        conf,
        args.repub_key,
        args.sub_key,
        args.add_matching_listener,
    )
