# List of useful network tools and commands for Zenoh users
This document provides a collection of network tools and commands that can be useful when working with Zenoh, a data-centric communication middleware. These tools can help you monitor, debug, and optimize your Zenoh network setup.

## Network Diagnostics

### Check IP Addresses and Network Interfaces
```bash
# Show all IP addresses on the machine
hostname -I

# Show detailed network interface information
ip addr show

# Show only IPv4 addresses
ip addr show | grep inet

# Show only IPv6 addresses
ip -6 addr show | grep inet6
```

### Check Network Routes
```bash
# Show routing table
ip route show

# Trace route to a destination (ICMP)
traceroute <IP_ADDRESS>

# Trace route using TCP packets (useful when ICMP is blocked)
traceroute -T -p <PORT> <IP_ADDRESS>

# Trace route using ICMP explicitly
traceroute -I <IP_ADDRESS>
```

### Test Network Connectivity
```bash
# Ping a host
ping <IP_ADDRESS>

# Ping with specific packet size (for MTU testing)
ping -M do -s 1400 <IP_ADDRESS>

# Test TCP port connectivity with netcat
nc -zv <IP_ADDRESS> <PORT>

# Test TCP port connectivity with telnet
telnet <IP_ADDRESS> <PORT>
```

## Zenoh-Specific Commands

### Starting Zenoh Router (zenohd)
```bash
# Start zenohd with default settings
zenohd

# Start zenohd listening on all IPv4 interfaces
zenohd --listen tcp/0.0.0.0:7447

# Start zenohd listening on specific IP
zenohd --listen tcp/<IP_ADDRESS>:7447

# Start zenohd on localhost only
zenohd --listen tcp/127.0.0.1:7447

# Start zenohd on a different port
zenohd --listen tcp/0.0.0.0:8888
```

### Running Zenoh Python Examples

#### Publisher
```bash
# Publisher in peer mode
python3 examples/z_pub.py

# Publisher connecting to specific endpoint
python3 examples/z_pub.py -e tcp/<IP_ADDRESS>:7447

# Publisher in client mode connecting to router
python3 examples/z_pub.py -m client -e tcp/<IP_ADDRESS>:7447

# Publisher in peer mode with specific listen address
python3 examples/z_pub.py -m peer -l tcp/0.0.0.0:7447
```

#### Subscriber
```bash
# Subscriber in peer mode
python3 examples/z_sub.py

# Subscriber connecting to specific endpoint
python3 examples/z_sub.py -e tcp/<IP_ADDRESS>:7447

# Subscriber in client mode connecting to router
python3 examples/z_sub.py -m client -e tcp/<IP_ADDRESS>:7447

# Subscriber in peer mode with specific listen address
python3 examples/z_sub.py -m peer -l tcp/0.0.0.0:7447
```

### Check Zenoh Processes and Ports
```bash
# Check if zenohd is running
ps aux | grep zenoh

# Check what's listening on Zenoh's default port (7447)
netstat -tuln | grep 7447
sudo ss -tulnp | grep 7447

# Kill all zenohd processes
pkill -9 zenohd

# Kill all Python processes (use with caution!)
pkill -9 python3
```

## Firewall Configuration

### iptables (Linux Firewall)
```bash
# Check firewall rules
sudo iptables -L -n -v

# Allow incoming TCP connections on port 7447
sudo iptables -I INPUT -p tcp --dport 7447 -j ACCEPT

# Allow outgoing TCP connections on port 7447
sudo iptables -I OUTPUT -p tcp --sport 7447 -j ACCEPT

# Flush all firewall rules (WARNING: removes all rules)
sudo iptables -F
```

## Network Routing

### Managing Routes
```bash
# Add a specific host route (bypass VPN or other routes)
sudo ip route add <IP_ADDRESS>/32 via <GATEWAY_IP> dev <INTERFACE>

# Example: Route specific IP through local gateway
sudo ip route add 172.26.191.65/32 via 192.168.0.1 dev wlp3s0

# Delete a route
sudo ip route del <NETWORK/CIDR> via <GATEWAY_IP> dev <INTERFACE>

# Add route with specific metric (priority)
sudo ip route add <IP_ADDRESS>/32 via <GATEWAY_IP> dev <INTERFACE> metric 50
```

## SSH Tunneling (for restricted networks)

### Reverse Tunnel
When you can SSH to a remote machine but can't directly connect to its ports:
```bash
# Create reverse tunnel (makes remote port accessible on local machine)
ssh -R <LOCAL_PORT>:localhost:<REMOTE_PORT> user@<REMOTE_IP>

# Example: Forward remote port 7447 to local port 7447
ssh -R 7447:localhost:7447 xilinx@172.26.191.65

# Run in background
ssh -fN -R 7447:localhost:7447 xilinx@172.26.191.65
```

**Usage after creating reverse tunnel:**
1. On remote machine: `zenohd --listen tcp/127.0.0.1:7447`
2. On local machine: `python3 examples/z_pub.py -m client -e tcp/127.0.0.1:7447`

### Forward Tunnel
When you want to access a remote port through SSH:
```bash
# Create forward tunnel (makes remote port accessible locally)
ssh -L <LOCAL_PORT>:localhost:<REMOTE_PORT> user@<REMOTE_IP>

# Example: Forward remote port 7447 to local port 7447
ssh -L 7447:localhost:7447 xilinx@172.26.191.65

# Run in background
ssh -fN -L 7447:localhost:7447 xilinx@172.26.191.65
```

## WiFi Management (NetworkManager)
```bash
# List available WiFi networks
nmcli device wifi list

# Connect to WiFi network
nmcli device wifi connect "<SSID>" password "<PASSWORD>"

# Show current connection status
nmcli connection show

# Show device status
nmcli device status
```

## Troubleshooting Tips

### Common Issues and Solutions

1. **"Address already in use" error**
   - Kill existing processes: `pkill -9 zenohd`
   - Check what's using the port: `sudo ss -tulnp | grep 7447`
   - Use a different port: `zenohd --listen tcp/0.0.0.0:7448`

2. **Ping works but TCP doesn't**
   - Check routing: `traceroute -T -p 7447 <IP_ADDRESS>`
   - Add specific route: `sudo ip route add <IP>/32 via <GATEWAY> dev <INTERFACE>`
   - Try SSH tunnel as workaround

3. **Can't connect between machines on different subnets**
   - Ensure both on same network: Check `hostname -I` on both
   - Check routing table: `ip route show`
   - Connect to same WiFi or use Ethernet cable

4. **IPv6 vs IPv4 issues**
   - If listening shows `:::7447`, it's IPv6
   - Force IPv4: `zenohd --listen tcp/0.0.0.0:7447`
   - Check with: `sudo ss -tulnp | grep 7447`

5. **VPN interfering with local connections**
   - Check routes: `ip route show`
   - Add specific route to bypass VPN: `sudo ip route add <IP>/32 via <LOCAL_GATEWAY> dev <INTERFACE> metric 50`
   - Use SSH tunnel as alternative
