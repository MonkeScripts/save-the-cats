# Zenoh setup 

This project uses Zenoh for communication between devices. Follow the steps below to set up Zenoh on your system.

## Zenoh Pico setup
The platformio configuration is already in this repo. 
1. Install platformio extension in Vscode.
2. Add your own `secrets.h` file in `include` directory. This file would include: 
    - Wifi SSID
    - Wifi password
    - Root CA cert in base64 (Cert generation would be explained in detail later)
You can refer to the `dummy_secrets.h` to see what needs to be there.

> Note: We are working on a forked version of Zenoh-pico. This is because they do not offer TLS communication for embedded devices. I have edited their source code to enable TLS communcation between the firebeetle and our computer. See this [PR](https://github.com/eclipse-zenoh/zenoh-pico/pull/1152) and its accompanying issue for better context.

## Setting up on linux computer
Current setup on Ubuntu 22.04 LTS
### Zenoh router installation
Adapted from https://zenoh.io/docs/getting-started/installation/

1. Add Eclipse Zenoh public key to apt keyring:
    ```bash
    curl -L https://download.eclipse.org/zenoh/debian-repo/zenoh-public-key | sudo gpg --dearmor --yes --output /etc/apt/keyrings/zenoh-public-key.gpg
    ```
2. Add Eclipse Zenoh private repository to the sources list:
    ```bash
    echo "deb [signed-by=/etc/apt/keyrings/zenoh-public-key.gpg] https://download.eclipse.org/zenoh/debian-repo/ /" | sudo tee -a /etc/apt/sources.list > /dev/null
    sudo apt update
    ```
3. Install zenoh-router package:
    ```bash
    sudo apt install zenoh
    ```
4. Then you can start the Zenoh router with this command:
    ```bash
    zenohd
    ``` 
### Zenoh python installation
Adapted from: https://github.com/eclipse-zenoh/zenoh-python

Set up [rust](https://doc.rust-lang.org/cargo/getting-started/installation.html) (if not already installed):
```bash
curl https://sh.rustup.rs -sSf | sh
source $HOME/.cargo/env
```
Clone our fork: https://github.com/MonkeScripts/zenoh-python
```bash
git clone https://github.com/MonkeScripts/zenoh-python
```  
In the cloned directory:

Using a virtual environment is strongly recommended to avoid Python version conflicts and dependency issues.

Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```     
Install development requirements:
```bash
pip install -r requirements-dev.txt
```
Build and install in development mode:
```bash

maturin develop --release
```
### Using tmuxp
We use tmuxp to manage multiple tmux sessions. Install it via:
```bash
sudo apt install tmuxp
```
To start a tmux session, use:
```bash
tmuxp load <path-to-this-repo>/tmuxp/comms.yaml
```
### Zenoh mqtt bridge installation
Adapted from: https://github.com/eclipse-zenoh/zenoh-plugin-mqtt
Since we already have the keyrings and sources list set up from the zenoh installation, we can directly install the zenoh-bridge-mqtt package:
```bash
sudo apt install zenoh-bridge-mqtt
```

### Using mqtt explorer (mosquitto)
To install mosquitto broker, download the deban package from https://mqtt-explorer.com/ and install it via:
```bash
sudo dpkg -i <deb-package-file>
```
You should then be able to see the mqtt explorer icon in your Applications.


## Generating SSL/TLS certificates
Adapted from: https://zenoh.io/docs/manual/tls/
To enable secure communication using SSL/TLS, you need to generate the necessary certificates. We use minica to generate our certificates.
First, install the [Go tools](https://golang.org/dl/) and set up your $GOPATH. Then, run:
```bash
go install github.com/jsha/minica@latest
```
Two certificates are needed — one for the local computer's TLS listener and one for the Ultra96's Tailscale IP:

**Local computer cert** (TLS listener on local machine):
```bash
~/go/bin/minica --ip-addresses 127.0.0.1
```

**Ultra96 cert** (TLS listener on Ultra96 over Tailscale — replace `100.125.252.109` with the actual Tailscale IP if it changes):
```bash
~/go/bin/minica --ip-addresses 100.125.252.109
```

On first run, minica will generate a keypair and a root certificate in the current directory, and will reuse that same keypair and root certificate unless they are deleted.

On each run, minica will generate a new keypair and sign an end-entity (leaf) certificate for that keypair. The certificate will contain a list of DNS names and/or IP addresses from the command line flags. The key and certificate are placed in a new directory whose name is chosen as the first domain name from the certificate, or the first IP address if no domain names are present. It will not overwrite existing keys or certificates.

The certificate will have a validity of 2 years and 30 days.

After generating the certificates, you should expect the following files:
- `minica.pem`: The root CA
- `minica-key.pem`: The root CA key

In each IP-specific folder (e.g. `127.0.0.1/` and `100.125.252.109/`):
- `cert.pem`: Server side certificate
- `key.pem`: Server side key

Please add the root CA certificate into your `secrets.h` in the `include` folder.
Please add the paths to these files in `local_computer/ROUTER_CONFIG.json5`, `local_computer/SESSION_CONFIG.json5`, `local_computer/BRIDGE_CONFIG.json5`, `ultra96/ROUTER_CONFIG.json5`, and `ultra96/SESSION_CONFIG.json5`.
>Note that `cert.pem` and `key.pem` are need for the MQTT explorer as well as the Unity program as well for MQTTS communication

### Using MQTT explorer
To enable MQTTS (MQTT + TLS), we need to add our own certificates into the application (Server side certificate and Server side key)
1. Open the MQTT explorer and under the `advanced` setting portion, add `cert.pem` and `key.pem` accordingly (You should already have generated the certificates based on the previous step).
2. Change the port to `8883` and toggle the TLS option
You should be able to see the topics streaming in.

## Using the Ultra96 board
### Setting up SSH keys
Refer to this https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server to setup ssh keys. This is useful if you do not want to type in the password every time when you ssh in.

### Current Approach: Tailscale
The Ultra96 is accessed directly over Tailscale, which gives it a stable IP without requiring a VPN or reverse tunnel.

1. Install Tailscale on both your local computer and the Ultra96:
    ```bash
    curl -fsSL https://tailscale.com/install.sh | sh
    ```
2. On each device, authenticate:
    ```bash
    sudo tailscale up
    ```
3. Find the Ultra96's Tailscale IP:
    ```bash
    tailscale ip -4
    ```
    (Current IP: `100.125.252.109`)

4. Generate a TLS cert for the Ultra96's Tailscale IP using minica (run on your local machine where minica and the root CA keys are):
    ```bash
    ~/go/bin/minica --ip-addresses 100.125.252.109
    ```
    This creates `100.125.252.109/cert.pem` and `100.125.252.109/key.pem`.

5. Copy the TLS directory to the Ultra96 (placing it at `/home/xilinx/save-the-cats/tls/`):
    ```bash
    scp -r 100.125.252.109 xilinx@100.125.252.109:/home/xilinx/save-the-cats/tls/
    # Also copy minica.pem (root CA) if not already there
    scp minica.pem xilinx@100.125.252.109:/home/xilinx/save-the-cats/tls/
    ```

6. Verify that `ultra96/ROUTER_CONFIG.json5` and `ultra96/SESSION_CONFIG.json5` point to the correct Tailscale IP cert paths. Update `local_computer/ROUTER_CONFIG.json5` `connect.endpoints` to use the correct Tailscale IP.

7. SSH into the Ultra96 directly via Tailscale (no tunnel needed):
    ```bash
    ssh xilinx@100.125.252.109
    ```

8. Once logged into the Ultra96, you should be in the dedicated python virtual environment: `pynq-venv`. This is because the script to activate the virtual environment is already configured in the `/etc/profile.d/pynq_venv.sh`. **Note that zenoh python package is already installed in this virtual environment**.
 If not in the environment, activate it by running:
    ```bash
    source /usr/local/share/pynq-venv/bin/activate
    ```

### Previous Approach: Reverse SSH Tunnel (SoC VPN / FortiClient)
> Note: This approach required SoC VPN (FortiClient) to be active. It has been replaced by Tailscale.
>
> **Setting up SoC VPN**: Install the debian for **fortinet_vpn** only. The other fortinet debians require an endpoint management system (which we do not have).

Set up a reverse ssh tunnel because the Ultra96 is behind the school's firewall.
1. On your local computer, run:
    ```bash
    ssh -R 7448:127.0.0.1:7448 xilinx@makerslab-fpga-22.ddns.comp.nus.edu.sg
    ```
    This maps port 7448 on the remote Ultra96 to port 7448 on your local computer.

    **Explanation of the parameters**:

    -R 7448: The port on the Remote Server that will be opened.

    127.0.0.1:7448	Where the traffic should go once it reaches your local machine (localhost, port 7448).

2. Once logged into the Ultra96, you should be in the dedicated python virtual environment: `pynq-venv`. This is because the script to activate the virtual environment is already configured in the `/etc/profile.d/pynq_venv.sh`. **Note that zenoh python package is already installed in this virtual environment**.
 If not in the environment, activate it by running:
    ```bash
    source /usr/local/share/pynq-venv/bin/activate
    ```
3. In a tmux session, start an example publisher or subscriber to test the setup.
    ```bash
    python3 examples/z_pub.py -e tcp/127.0.0.1:7448
    ```
   Note: Please do not use `localhost`, it might be interpreted as ipv6 version instead
    You can check whether the port is open by running:
    ```bash
    sudo netstat -nlp | grep 7448
    ```
    <img width="1101" height="107" alt="image" src="https://github.com/user-attachments/assets/b146dd6c-590b-4421-8a25-1f2be67b878d" />

## Using Config files
We use config files to simplify the connection setup. Each device has a `ROUTER_CONFIG.json5` and a `SESSION_CONFIG.json5` file in the our Zenoh Python fork.
To run the examples with config files, use the following commands:
For Routers:
```bash
zenohd -c <path-to-this-repo>/zenoh/configs/<device>/ROUTER_CONFIG.json5
```
For Publishers/Subscribers:
```bash
python3 zenoh_scripts/z_pub.py -c <path-to-this-repo>/zenoh/configs/<device>/SESSION_CONFIG.json5
```

## Tests --WIP--

### Testing Ultra96 connection
In a tmux, using the config files:
1. On the Ultra96, start the Zenoh router:
    ```bash
    zenohd -c <path-to-repo>/zenoh/configs/ultra96/ROUTER_CONFIG.json5
    ```
2. On your local computer, start the Zenoh router (it will connect to the Ultra96 via Tailscale):
    ```bash
    zenohd -c <path-to-repo>/zenoh/configs/local_computer/ROUTER_CONFIG.json5
    ```
3. On your computer, run the Zenoh subscriber example:
    ```bash
    (.venv) python3 zenoh_scripts/z_sub.py -c <path-to-repo>/zenoh/configs/local_computer/SESSION_CONFIG.json5
    ```
    You should then be able to see messages being published from the Ultra96 board.

## Testing the setup (Esp32 and local computer)
1. Start the zenoh router on your computer:
    ```bash
    zenohd
    ```
2. On your ESP32/FireBeetle device, upload and run the Zenoh publisher example.
3. On your computer, activate the virtual environment and run the Zenoh publisher example from the `examples/z_pub.py`
    ```bash
    (.venv) python3 examples/z_pub.py -k demo/example/test -p 'Hello World'
    ```
4. On another terminal, run the Zenoh subscriber example from the `examples/z_sub.py`
    ```bash
    (.venv) python3 examples/z_sub.py -k 'demo/**'
    ```
5. You can check the zenoh config by running:
    ```bash
    (.venv) python3 examples/z_sub.py -k 'demo/**'
    ```

You should see messages being published from both the ESP32/FireBeetle and your computer.
Here is an image:
<img width="2560" height="1600" alt="image" src="https://github.com/user-attachments/assets/f52007e3-7214-4273-9919-3814f6f87a13" />

## Zenoh + InfluxDB 2.x Setup

### 1. Install InfluxDB 2.x (Ubuntu/Debian)

Add the repository and install:

```bash
# influxdata-archive.key GPG fingerprint:
#   Primary key fingerprint: 24C9 75CB A61A 024E E1B6  3178 7C3D 5715 9FC2 F927
#   Subkey fingerprint:      9D53 9D90 D332 8DC7 D6C8  D3B9 D8FF 8E1F 7DF8 B07E
wget -q https://repos.influxdata.com/influxdata-archive.key
gpg --show-keys --with-fingerprint --with-colons ./influxdata-archive.key 2>&1 | grep -q '^fpr:\+24C975CBA61A024EE1B631787C3D57159FC2F927:' \
  && cat influxdata-archive.key | gpg --dearmor \
  | sudo tee /etc/apt/keyrings/influxdata-archive.gpg > /dev/null

echo 'deb [signed-by=/etc/apt/keyrings/influxdata-archive.gpg] https://repos.influxdata.com/debian stable main' \
  | sudo tee /etc/apt/sources.list.d/influxdata.list

sudo apt update
sudo apt install influxdb2 -y
```

Start the service:
```bash
sudo systemctl start influxdb
sudo systemctl enable influxdb
```

Initial setup:
```bash
influx setup
```
You will be prompted for: username, password, organization name, bucket name, and retention period.

Get your credentials:
```bash
influx org list          # Copy the hex ID (e.g. a51e53a6b98d9c2e)
influx auth list --json  # Copy the full token string
```

### 2. Install the Zenoh InfluxDB v2 Backend Plugin

Add the zenoh repo (if not already added):
```bash
echo "deb [trusted=yes] https://download.eclipse.org/zenoh/debian-repo/ /" \
  | sudo tee -a /etc/apt/sources.list.d/zenoh.list > /dev/null
sudo apt update
```

Install the plugin:
```bash
sudo apt install zenoh-backend-influxdb-v2
```

**Important: Rust version compatibility**

Zenoh requires the plugin to be built with the exact same Rust compiler version as `zenohd`. Check with:
```bash
zenohd --version
```

If you see a mismatch error like `Incompatible rustc versions`, you need to build the plugin from source:
```bash
# Install Rust and set the version to match zenohd
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env
rustup install <rustc_version_from_zenohd>
rustup default <rustc_version_from_zenohd>
rustup toolchain  list  # Verify the correct version is active

# Clone and build
git clone https://github.com/eclipse-zenoh/zenoh-backend-influxdb.git
cd zenoh-backend-influxdb
git checkout <your_zenoh_version>   # e.g. 1.8.0
cargo build +<rustc_version_from_zenohd> -p zenoh-backend-influxdb-v2

# Replace the system .so file
sudo cp target/release/libzenoh_backend_influxdb2.so /usr/lib/libzenoh_backend_influxdb2.so
```

### 3. Visualize in InfluxDB UI

1. Open `http://localhost:8086` in your browser
2. Go to **Data Explorer**
3. Select your bucket (e.g. `example`)
4. Select the measurement (your zenoh key)
5. Select `value` under `_field`
6. Click **SUBMIT**

**Plotting numeric data as a graph**

Use the Script Editor with this Flux query:
```flux
from(bucket: "example")
  |> range(start: -1h)
  |> filter(fn: (r) => r["_field"] == "value")
  |> filter(fn: (r) => r["kind"] == "PUT")
  |> toFloat()
  |> aggregateWindow(every: 1s, fn: last, createEmpty: false)
  |> yield(name: "zenoh_data")
```

> `toFloat()` only works if the stored values are plain numbers. If you stored non-numeric strings, delete the old data first:
> ```bash
> influx bucket delete --name example --org <your-org>
> influx bucket create --name example --org <your-org>
> ```

### Quick Reference

| Command | Purpose |
|---------|---------|
| `influx setup` | Initial InfluxDB setup |
| `influx org list` | Get org hex ID |
| `influx auth list --json` | Get full API token |
| `influx bucket list` | List all buckets |
| `influx bucket delete --name X --org Y` | Delete a bucket |
| `influx bucket create --name X --org Y` | Create a bucket |
