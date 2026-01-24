# Zenoh setup 

This project uses Zenoh for communication between devices. Follow the steps below to set up Zenoh on your system.

## Zenoh Pico setup
Adapted from [here](https://github.com/eclipse-zenoh/zenoh-pico)

Include the following line in platformio.ini:
```
lib_deps = https://github.com/eclipse-zenoh/zenoh-pico
```

## Setting up on linux computer
Current setup on Ubuntu 22.04 LTS
### Zenoh router installation
Adapted from https://zenoh.io/docs/getting-started/installation/

1. Add Eclipse Zenoh public key to apt keyring
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
Clone the repository: https://github.com/eclipse-zenoh/zenoh-python
```bash
git clone https://github.com/eclipse-zenoh/zenoh-python
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
## Using the Ultra96 board
### Setting up ssh keys
Refer to this https://www.digitalocean.com/community/tutorials/how-to-configure-ssh-key-based-authentication-on-a-linux-server to setup ssh keys. This is useful if you do not want to type in the password every time when you ssh in.

### Accessing the Ultra96 board
Set up a reverse ssh tunnel because the Ultra96 is behind the school's firewall.
1. On your local computer, run:
    ```bash
    ssh -R 7448:127.0.0.1:7448 xilinx@makerslab-fpga-22.ddns.comp.nus.edu.sg
    ```
    This maps port 7448 on the remote Ultra96 to port 7448 on your local computer.

    **Explanation of the parameters**:

    -R 7448: The port on the Remote Server that will be opened.

    127.0.0.1:7448	Where the traffic should go once it reaches your local machine (localhost, port 7448).

2. Once logged into the Ultra96, you should be in the dedicated python virtual environment: `pynq-venv`. This is because the script to activate the virtual environment is already configured in the `/etc/profile.d/pynq_venv.sh`. **Note that zenoh python package is already installed in this virtual environment. Setup is exactly the same as setting up zenoh on a linux computer**.
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

### Local computer changes for Ultra96 connection
In a tmux:
1. Start the zenoh router on your computer, binding to port 7448:
    ```bash
    zenohd -l tcp/[::]:7448
    ```
2. On your computer, activate the virtual environment and run the Zenoh subscriber example from the `examples/z_sub.py`, connecting to the Ultra96:
    ```bash
    (.venv) python3 examples/z_sub.py -e tcp/127.0.0.1:7448
    ```
    You should be then able to see messages being published from the Ultra96 board.

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
