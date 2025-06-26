#!/bin/bash

# Configure API settings first
echo "Configuring IBKR Gateway API settings..."
/opt/ibkr/scripts/configure-api.sh

# Start virtual display
echo "Starting virtual display..."
Xvfb :1 -screen 0 1024x768x24 &
export DISPLAY=:1

# Wait for X server to start
sleep 2

# Start window manager
echo "Starting window manager..."
fluxbox &

# Start VNC server for remote access
echo "Starting VNC server..."
x11vnc -display :1 -nopw -listen localhost -xkb -ncache 10 -ncache_cr -forever &

# Wait a bit for services to start
sleep 3

# Start IBKR Gateway
echo "Starting IBKR Gateway..."
echo "Configuration:"
echo "  Host: $IBKR_HOST"
echo "  Port: $IBKR_PORT"
echo "  Client ID: $IBKR_CLIENT_ID"
echo "  Account: $ACCOUNT_CODE"
echo "  API Mode: Read-Write (Trading Enabled)"
echo "  Trusted IPs: All IPs allowed (0.0.0.0)"

cd /opt/ibkr/IBGateway

# Start the gateway with proper JVM options
./ibgateway -J-Xmx1024m -J-Djava.awt.headless=false &

# Keep the container running and show logs
echo "IBKR Gateway container is running..."
echo "VNC access available on port 5900"
echo "API access available on port $IBKR_PORT"

# Monitor the gateway process
GATEWAY_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down IBKR Gateway..."
    kill $GATEWAY_PID 2>/dev/null
    pkill -f ibgateway
    pkill -f Xvfb
    pkill -f x11vnc
    pkill -f fluxbox
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Wait for the gateway process
wait $GATEWAY_PID

# If gateway exits, keep container alive for debugging
echo "IBKR Gateway process ended. Keeping container alive for debugging..."
while true; do
    sleep 60
    echo "Container still running... ($(date))"
done

