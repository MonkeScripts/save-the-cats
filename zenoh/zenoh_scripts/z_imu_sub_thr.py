import time
import json
import zenoh
from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber


def format_bytes(size):
    """Formats bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def main(conf: zenoh.Config, key: str, measure_count: int):
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    print("Opening session...")
    with zenoh.open(conf) as session:
        print(f"Subscribing to '{key}'...")

        state = {
            "total_count": 0,
            "total_bytes": 0,
            "batch_count": 0,
            "batch_bytes": 0,
            "start_time": None,
            "global_start": None,
        }

        def listener(sample: zenoh.Sample):
            now = time.time()
            if state["global_start"] is None:
                state["global_start"] = now
                state["start_time"] = now

            # Track counts and payload size
            payload_size = len(sample.payload)
            state["total_count"] += 1
            state["total_bytes"] += payload_size
            state["batch_count"] += 1
            state["batch_bytes"] += payload_size
            timestamp_str = (
                sample.timestamp.to_string_rfc3339_lossy()
                if sample.timestamp
                else "N/A"
            )
            print(
                f">> [Subscriber] Received {sample.kind} at {timestamp_str} ('{sample.key_expr}': '{sample.payload.to_string()}')"
            )
            data = json.loads(sample.payload.to_string())
            print(f"Json output: {data} ")

            if state["batch_count"] >= measure_count:
                elapsed = now - state["start_time"]
                if elapsed > 0:
                    msg_thr = state["batch_count"] / elapsed
                    byte_thr = state["batch_bytes"] / elapsed

                    print(f"--- Statistics (Last {measure_count} msgs) ---")
                    print(f"  Throughput: {msg_thr:.2f} msgs/s")
                    print(f"  Bandwidth:  {format_bytes(byte_thr)}/s")
                    print(
                        f"  Total Rcvd: {state['total_count']} msgs ({format_bytes(state['total_bytes'])})"
                    )

                state["batch_count"] = 0
                state["batch_bytes"] = 0
                state["start_time"] = now

        advanced_sub = declare_advanced_subscriber(
            session,
            key,
            listener,
            history=HistoryConfig(detect_late_publishers=True),
            recovery=RecoveryConfig(heartbeat=True),
            subscriber_detection=True,
        )

        def miss_listener(miss: Miss):
            print(f"!! Missed {miss.nb} samples from {miss.source}")

        advanced_sub.sample_miss_listener(miss_listener)
        try:
            while True:
                time.sleep(0.5)  # Print every 5 seconds
                if state["global_start"]:
                    total_elapsed = time.time() - state["global_start"]
                    if total_elapsed > 0:
                        avg_throughput = state["total_count"] / total_elapsed
                        avg_bandwidth = state["total_bytes"] / total_elapsed
                        print(f"\n--- Average (since start) ---")
                        print(f"Avg Throughput: {avg_throughput:.2f} msgs/s")
                        print(f"Avg Bandwidth:  {format_bytes(avg_bandwidth)}/s")
        except KeyboardInterrupt:
            print("\nShutdown initiated")


# --- Command line argument parsing ---
if __name__ == "__main__":
    import argparse
    import os
    import sys

    import common

    parser = argparse.ArgumentParser(prog="z_imu_sub_thr")
    common.add_config_arguments(parser)
    parser.add_argument("--key", "-k", default="esp/**", type=str)
    parser.add_argument(
        "--number", "-n", default=100, type=int, help="Batch size for logs"
    )

    args = parser.parse_args()
    main(common.get_config_from_args(args), args.key, args.number)
