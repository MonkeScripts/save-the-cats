import sys
import tty
import termios
import zenoh
import json
import random
import itertools
from zenoh.ext import CacheConfig, MissDetectionConfig, declare_advanced_publisher


def get_char():
    """Reads a single character from the terminal without pressing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main(conf: zenoh.Config, key: str, history: int):
    zenoh.init_log_from_env_or("error")

    print("Opening session...")
    with zenoh.open(conf) as session:
        print(f"Declaring AdvancedPublisher on '{key}'...")

        # Configure the publisher
        pub = declare_advanced_publisher(
            session,
            key,
            cache=CacheConfig(max_samples=history),
            sample_miss_detection=MissDetectionConfig(heartbeat=5),
            publisher_detection=True,
        )

        counter = itertools.count()

        print("\n--- Relay Active ---")
        print("TAP ANY KEY to publish a dummy packet.")
        print("Press 'q' or 'ESC' to exit.\n")

        try:
            while True:
                char = get_char()

                if char == "q" or char == "\x1b" or char == "\x03":
                    print("\nExiting...")
                    break

                idx = next(counter)

                data = {
                    "action": round(random.uniform(0, 3)),
                    "key_pressed": char,  # Debugging help
                }

                payload = json.dumps(data).encode("utf-8")
                pub.put(payload)

                print(f"[{idx}] Sent Action {data['action']} (Key: '{char}')")

        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    import argparse
    import common

    # Ensure standard libraries are available
    import os

    parser = argparse.ArgumentParser(
        prog="z_relay_pub", description="Zenoh manual keystroke publisher"
    )
    common.add_config_arguments(parser)
    parser.add_argument("--key", "-k", dest="key", default="computer/action1", type=str)
    parser.add_argument("--history", dest="history", type=int, default=1)

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(conf, args.key, args.history)
