import time
import hashlib
import zenoh
import argparse
import os
import ctypes
import platform

SAVE_FOLDER = "received_images"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)


def format_bytes(size):
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def set_wallpaper(image_path):
    """Sets the desktop wallpaper based on the Operating System."""
    abs_path = os.path.abspath(image_path)
    print(f">> Attempting to set wallpaper: {abs_path}")

    try:
        sys_name = platform.system()
        if sys_name == "Windows":
            # SPI_SETDESKWALLPAPER = 20
            ctypes.windll.user32.SystemParametersInfoW(20, 0, abs_path, 3)
        elif sys_name == "Darwin":  # macOS
            script = f'tell application "Finder" to set desktop picture to POSIX file "{abs_path}"'
            os.system(f"osascript -e '{script}'")
        else:  # Linux (Gnome)
            os.system(
                f"gsettings set org.gnome.desktop.background picture-uri file://{abs_path}"
            )
        print("✅ Wallpaper updated successfully!")
    except Exception as e:
        print(f"❌ Failed to set wallpaper: {e}")


def main(conf: zenoh.Config, selector: str, target: zenoh.QueryTarget, timeout: float):
    zenoh.init_log_from_env_or("error")
    print(f"Current Config: {conf}")

    with zenoh.open(conf) as session:
        query_selector = zenoh.Selector(selector)
        querier = session.declare_querier(
            query_selector.key_expr, target=target, timeout=timeout
        )

        sha256_hash = hashlib.sha256()
        print(f"Requesting image via Zenoh...")
        replies = querier.get(
            encoding=zenoh.Encoding.TEXT_JSON5, parameters=query_selector.parameters
        )
        for reply in replies:
            if reply.ok:
                print(f"REPLY OK")
                received_data = reply.ok.payload.to_bytes()
                sha256_hash.update(received_data)
                file_hash = sha256_hash.hexdigest()
                print(f"Received file_hash {file_hash}")
                filename = "wallpaper_update.jpg"
                full_path = os.path.join(SAVE_FOLDER, filename)

                with open(full_path, "wb") as f:
                    f.write(received_data)

                # Change the wallpaper
                set_wallpaper(full_path)
            else:
                print(f"REPLY NOT OK")


if __name__ == "__main__":
    import common

    parser = argparse.ArgumentParser()
    common.add_config_arguments(parser)
    parser.add_argument("--selector", "-s", default="BIG/**", type=str)
    args = parser.parse_args()
    main(
        common.get_config_from_args(args),
        args.selector,
        zenoh.QueryTarget.ALL,
        30.0,
    )
