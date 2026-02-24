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
from typing import Optional

import zenoh
from zenoh import ZBytes
from zenoh.ext import CacheConfig, MissDetectionConfig, declare_advanced_publisher


def main(conf: zenoh.Config, key: str, history: int):
    # initiate logging
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

        print("Press CTRL-C to quit...")
        for idx in itertools.count():

            ax, ay, az, gx, gy, gz = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            imu1 = {
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "action": 0,
            }
            payload_string: str = json.dumps(imu1)
            # Adapted from https://github.com/eclipse-zenoh/zenoh-python/blob/1.0.0-beta.4/examples/z_bytes.py
            assert payload_string is not None
            payload_bytes = ZBytes(payload_string)
            time.sleep(1)
            print(f"Putting Data ('{key}': '{payload_bytes}')... index: {idx}")
            pub.put(payload_bytes)


# --- Command line argument parsing --- --- --- --- --- ---
if __name__ == "__main__":
    import argparse
    import itertools

    import common
    import json

    parser = argparse.ArgumentParser(
        prog="z_advanced_pub", description="zenoh advanced pub example"
    )
    common.add_config_arguments(parser)
    parser.add_argument(
        "--key",
        "-k",
        dest="key",
        default="esp/imu1",
        type=str,
        help="The key expression to publish onto.",
    )
    parser.add_argument(
        "--history",
        dest="history",
        type=int,
        default=1,
        help="The number of publications to keep in cache",
    )

    args = parser.parse_args()
    conf = common.get_config_from_args(args)

    main(conf, args.key, args.history)
