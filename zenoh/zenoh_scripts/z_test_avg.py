import time
import zenoh
import threading
from typing import List


def main(
    conf: zenoh.Config,
    repub_key: str,
    sub_key: str,
    interval: float,
    add_matching_listener: bool,
):
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    # State for averaging
    data_buffer: List[float] = []
    buffer_lock = threading.Lock()

    print("Opening session...")
    with zenoh.open(conf) as session:

        # 1. Setup Publisher
        print(f"Declaring Publisher on '{repub_key}'...")
        pub = session.declare_publisher(repub_key)

        if add_matching_listener:

            def on_matching_status_update(status: zenoh.MatchingStatus):
                state = "has" if status.matching else "has NO MORE"
                print(f"Publisher {state} matching subscribers.")

            pub.declare_matching_listener(on_matching_status_update)

        # 2. Setup Subscriber (The "Enqueuer")
        def republish_callback(sample: zenoh.Sample):
            try:
                # Should give me gibberish values
                msg = sample.payload.to_string()
                val = float(msg[-1])
                with buffer_lock:
                    data_buffer.append(val)
                # Optional: print raw arrival for debugging
                print(f"Received: {val}")
            except ValueError:
                print(
                    f">> [Warning] Non-numeric data ignored: {sample.payload.to_string()}"
                )

        print(f"Declaring Subscriber on '{sub_key}'...")
        sub = session.declare_subscriber(sub_key, republish_callback)

        # 3. Main Loop (The "Averager")
        print(f"Averaging every {interval}s. Press CTRL-C to quit...")
        try:
            while True:
                time.sleep(interval)

                with buffer_lock:
                    if data_buffer:
                        avg_val = sum(data_buffer) / len(data_buffer)
                        count = len(data_buffer)
                        data_buffer.clear()  # Clear queue for next window

                        output = f"Average({count} samples): {avg_val:.2f}"
                        print(f">> [Repub] {output}")
                        pub.put(output)
                    else:
                        # Optional: Print if no data arrived in the window
                        pass

        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    # Compatibility logic for your local 'examples.common'
    _this_dir = os.path.dirname(__file__)
    _repo_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    try:
        import examples.common as common
    except ImportError:
        # Fallback if the examples folder isn't found
        common = None

    parser = argparse.ArgumentParser(
        prog="z_avg_repub", description="Zenoh Averaging Republisher"
    )

    if common:
        common.add_config_arguments(parser)

    parser.add_argument(
        "--key",
        "-k",
        dest="repub_key",
        default="demo/avg/output",
        help="Key to publish average onto.",
    )
    parser.add_argument(
        "--sub-key", "-s", default="demo/example/**", help="Key to subscribe to."
    )
    parser.add_argument(
        "--interval", "-i", type=float, default=5.0, help="Window interval in seconds."
    )
    parser.add_argument(
        "--add-matching-listener", action="store_true", help="Add matching listener"
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args) if common else zenoh.Config()

    main(conf, args.repub_key, args.sub_key, args.interval, args.add_matching_listener)
