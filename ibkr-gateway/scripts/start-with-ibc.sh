#!/bin/bash

# IBKR Gateway Startup Script with IBC Automation
# This script starts the Gateway with automated login using IBC

echo "=" * 60
echo "IBKR Gateway with IBC Automation"
echo "=" * 60

# Configure IBC with environment variables
echo "Configuring IBC with credentials..."

# Update IBC config with environment variables
sed -i "s/IbLoginId=.*/IbLoginId=${IBKR_USERNAME}/" /home/ibkr/ibc/config.ini
sed -i "s/IbPassword=.*/IbPassword=${IBKR_PASSWORD}/" /home/ibkr/ibc/config.ini
sed -i "s/ApiPortNumber=.*/ApiPortNumber=${IBKR_PORT}/" /home/ibkr/ibc/config.ini

# Set trading mode based on port
if [ "$IBKR_PORT" = "4001" ]; then
    sed -i "s/TradingMode=.*/TradingMode=live/" /home/ibkr/ibc/config.ini
    echo "Trading Mode: LIVE"
else
    sed -i "s/TradingMode=.*/TradingMode=paper/" /home/ibkr/ibc/config.ini
    echo "Trading Mode: PAPER"
fi

echo "Configuration:"
echo "  Username: ${IBKR_USERNAME}"
echo "  Password: [HIDDEN]"
echo "  Port: ${IBKR_PORT}"
echo "  Account: ${ACCOUNT_CODE}"
echo "  Trading Mode: $(grep TradingMode /home/ibkr/ibc/config.ini | cut -d'=' -f2)"

# Start virtual display
echo "Starting virtual display..."
Xvfb :1 -screen 0 1024x768x24 &
XVFB_PID=$!
export DISPLAY=:1

# Wait for X server to start
sleep 3

# Start window manager
echo "Starting window manager..."
fluxbox &
FLUXBOX_PID=$!

# Start VNC server for remote access
echo "Starting VNC server..."
x11vnc -display :1 -nopw -listen localhost -xkb -ncache 10 -ncache_cr -forever &
VNC_PID=$!

# Wait for services to start
sleep 3

# Start socat to relay API connections
echo "Starting API relay..."
socat TCP-LISTEN:${IBKR_PORT},bind=0.0.0.0,fork TCP:127.0.0.1:${IBKR_PORT} &
SOCAT_PID=$!

# Start IBKR Gateway with IBC
echo "Starting IBKR Gateway with IBC automation..."
echo "IBC will handle login automatically..."

cd /home/ibkr/ibc

# Set IBC environment variables
export TWS_MAJOR_VRSN=10
export TWS_PATH=/opt/ibkr/IBGateway
export IBC_INI=/home/ibkr/ibc/config.ini
export IBC_PATH=/home/ibkr/ibc
export JAVA_PATH=$JAVA_HOME/bin

# Start IBC with Gateway
./gatewaystart.sh &
IBC_PID=$!

echo "IBC started with PID: $IBC_PID"
echo "Gateway startup initiated..."

# Wait for API to become available
echo "Waiting for API to become available on port ${IBKR_PORT}..."

# Function to check if API port is listening
check_api_port() {
    netstat -tlnp 2>/dev/null | grep ":${IBKR_PORT}" > /dev/null
    return $?
}

# Wait up to 120 seconds for API to be ready
for i in {1..120}; do
    if check_api_port; then
        echo "✅ API is now available on port ${IBKR_PORT}!"
        break
    fi
    
    if [ $i -eq 120 ]; then
        echo "❌ API did not become available within 120 seconds"
        echo "Check logs for issues"
    else
        echo "⏳ Waiting for API... (${i}/120)"
        sleep 1
    fi
done

# Function to handle shutdown
cleanup() {
    echo "Shutting down IBKR Gateway and services..."
    kill $IBC_PID 2>/dev/null
    kill $SOCAT_PID 2>/dev/null
    kill $VNC_PID 2>/dev/null
    kill $FLUXBOX_PID 2>/dev/null
    kill $XVFB_PID 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

echo "IBKR Gateway container is running..."
echo "API access: ${IBKR_HOST}:${IBKR_PORT}"
echo "VNC access: localhost:5900"
echo "Press Ctrl+C to stop"

# Keep container running and monitor IBC process
while true; do
    if ! kill -0 $IBC_PID 2>/dev/null; then
        echo "IBC process ended, restarting..."
        cd /home/ibkr/ibc
        ./gatewaystart.sh &
        IBC_PID=$!
        echo "IBC restarted with PID: $IBC_PID"
    fi
    
    sleep 30
done

