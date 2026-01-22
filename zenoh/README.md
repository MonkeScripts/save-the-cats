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
When you're done, deactivate the virtual environment:
```bash

deactivate
```
## Testing the setup
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