# What to run / check for midpoint check
## Esp32
1. Change their topic names
2. Flash the proper code 
3. Plug into power

## Local computer
1. `tmuxp load <path-to-repo>/tmuxp/computer_demo.yaml`
2. Run mqtt explorer, connect to `127.0.0.1:8883` with TLS enabled

## Ultra96
1. Connect to SoC vpn
2. `ssh -R 7448:127.0.0.1:7448 xilinx@makerslab-fpga-22.ddns.comp.nus.edu.sg`
3. `tmux a` or tmuxp `load <path-to-repo>/tmuxp/ultra.yaml`

## Phone
1. Change IP to local computer's IP
2. Run the game