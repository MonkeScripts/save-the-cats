import time
import zenoh
from zenoh import ZBytes
from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber
import threading
from typing import List
import json


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

        def republish_callback(sample: zenoh.Sample):
            try:
                print(
                    f">> [Subscriber] Received {sample.kind} at {sample.timestamp.to_string_rfc3339_lossy()} ('{sample.key_expr}': '{sample.payload.to_string()}')"
                )
                msg = json.loads(sample.payload.to_string())
                print(msg)
                val = msg["action"]
                with buffer_lock:
                    data_buffer.append(val)
                print(f"Received: {val}")
            except ValueError:
                print(
                    f">> [Warning] Non-numeric data ignored: {sample.payload.to_string()}"
                )

        print(f"Declaring Subscriber on '{sub_key}'...")
        sub = declare_advanced_subscriber(
            session,
            sub_key,
            republish_callback,
            history=HistoryConfig(detect_late_publishers=True),
            recovery=RecoveryConfig(heartbeat=True),
            subscriber_detection=True,
        )

        def miss_listener(miss: Miss):
            print(f">> [Subscriber] Missed {miss.nb} samples from {miss.source} !!!")

        sub.sample_miss_listener(miss_listener)

        print(f"Averaging every {interval}s. Press CTRL-C to quit...")
        try:
            while True:
                time.sleep(interval)

                with buffer_lock:
                    if data_buffer:
                        avg_val = sum(data_buffer) // len(data_buffer)
                        data = {
                            "action": avg_val,
                        }
                        payload_string = json.dumps(data)
                        payload_bytes = ZBytes(payload_string)
                        pub.put(payload_bytes)
                        count = len(data_buffer)
                        print(
                            f">> [Repub] To Visualizer: average action after({count} samples): {avg_val:.2f}"
                        )
                        data_buffer.clear()  # Clear queue for next window
                    else:
                        # Optional: Print if no data arrived in the window
                        pass

        except KeyboardInterrupt:
            print("\nShutting down...")


if __name__ == "__main__":
    import argparse
    import os
    import sys

    import common

    parser = argparse.ArgumentParser(
        prog="z_repub", description="Zenoh Averaging Republisher"
    )

    if common:
        common.add_config_arguments(parser)

    parser.add_argument(
        "--key",
        "-k",
        dest="repub_key",
        default="ultra/action1",
        help="Key to publish average onto.",
    )
    parser.add_argument(
        "--sub-key", "-s", default="esp/**", help="Key to subscribe to."
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
