#
# Copyright (c) 2022 ZettaScale Technology
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
#
import time

import zenoh
from zenoh.ext import HistoryConfig, Miss, RecoveryConfig, declare_advanced_subscriber
import json


def main(conf: zenoh.Config, key: str):
    # initiate logging
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    print("Opening session...")
    with zenoh.open(conf) as session:
        print(f"Declaring Subscriber on '{key}'...")
        count = 0

        def listener(sample: zenoh.Sample):
            nonlocal count
            count += 1
            print(
                f">> [Subscriber] Received {sample.kind} at {sample.timestamp.to_string_rfc3339_lossy()} ('{sample.key_expr}': '{sample.payload.to_string()}, count: {count}')"
            )
            data = json.loads(sample.payload.to_string())
            print(f"Json output: {data} ")

        advanced_sub = declare_advanced_subscriber(
            session,
            key,
            listener,
            history=HistoryConfig(detect_late_publishers=True),
            recovery=RecoveryConfig(heartbeat=True),
            subscriber_detection=True,
        )

        def miss_listener(miss: Miss):
            print(f">> [Subscriber] Missed {miss.nb} samples from {miss.source} !!!")

        advanced_sub.sample_miss_listener(miss_listener)

        print("Press CTRL-C to quit...")
        while True:
            time.sleep(1)


# --- Command line argument parsing --- --- --- --- --- ---
if __name__ == "__main__":
    import argparse
    import os
    import sys

    _this_dir = os.path.dirname(__file__)
    _repo_root = os.path.abspath(os.path.join(_this_dir, ".."))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    import zenoh_scripts.common as common

    parser = argparse.ArgumentParser(
        prog="z_imu_sub", description="imu subscription from firebeetle"
    )
    common.add_config_arguments(parser)
    parser.add_argument(
        "--key",
        "-k",
        dest="key",
        default="esp/**",
        type=str,
        help="The key expression to subscribe to.",
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(conf, args.key)
