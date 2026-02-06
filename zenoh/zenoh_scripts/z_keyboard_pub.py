import zenoh
import json
import random
import itertools
from zenoh import ZBytes
from zenoh.ext import CacheConfig, MissDetectionConfig, declare_advanced_publisher

# We use pynput to capture keystrokes without needing to press "Enter"
# Install via: pip install pynput
from pynput import keyboard


def main(conf: zenoh.Config, key: str, history: int):
    zenoh.init_log_from_env_or("error")

    print("Opening session...")
    with zenoh.open(conf) as session:
        print(f"Declaring AdvancedPublisher on '{key}'...")
        pub = declare_advanced_publisher(
            session,
            key,
            cache=CacheConfig(max_samples=history),
            sample_miss_detection=MissDetectionConfig(heartbeat=5),
            publisher_detection=True,
        )

        counter = itertools.count()

        def on_press(key_event):
            try:
                idx = next(counter)

                # Generate random dummy IMU data
                data = {
                    "action": round(random.uniform(0, 3), 2),
                }

                payload_string = json.dumps(data)
                pub.put(payload_string)

                print(
                    f"[Key Pressed: {key_event}] Published to {key}with {data} | Index: {idx}"
                )

            except Exception as e:
                print(f"Error publishing: {e}")

        print("\n--- Relay Active ---")
        print("TAP ANY KEY to publish a dummy packet. Press ESC to exit.")

        # Start the keyboard listener
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()


if __name__ == "__main__":
    import argparse
    import common

    parser = argparse.ArgumentParser(
        prog="z_relay_pub", description="Zenoh manual keystroke publisher"
    )
    common.add_config_arguments(parser)
    parser.add_argument("--key", "-k", dest="key", default="computer/action1", type=str)
    parser.add_argument("--history", dest="history", type=int, default=1)

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(conf, args.key, args.history)
