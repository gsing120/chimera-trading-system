#!/bin/bash

# Configure IBKR Gateway API settings
echo "Configuring IBKR Gateway API settings..."

# Create the Jts directory if it doesn't exist
mkdir -p /home/ibkr/Jts

# Create API configuration file
cat > /home/ibkr/Jts/jts.ini << EOF
[IBGateway]
ApiOnly=true
ReadOnlyApi=false
TrustedIPs=0.0.0.0,127.0.0.1,::1,172.17.0.0/16,192.168.0.0/16,10.0.0.0/8
PortNumber=${IBKR_PORT}
EnableAPI=true
LogLevel=Information
LogComponents=never
LogToConsole=true
AcceptIncomingConnectionAction=accept
AllowBlindTrading=true
SendMarketDataInLotsOfHundred=false
DownloadOpenOrders=true
BypassOrderPrecautions=false

[Logon]
useRemoteSettings=false
tradingMode=paper
colorPaletteName=dark
Steps=8
Locale=en
UseSSL=true

[Communication]
Region=America
Peer=gdc1.ibllc.com:4001
ProxyHost=
ProxyPort=
ProxyUsername=
ProxyPassword=
UseProxy=false

[API]
EnableAPI=true
ReadOnlyApi=false
TrustedIPs=0.0.0.0,127.0.0.1,::1,172.17.0.0/16,192.168.0.0/16,10.0.0.0/8
PortNumber=${IBKR_PORT}
AcceptIncomingConnectionAction=accept
AllowBlindTrading=true
DownloadOpenOrders=true
BypassOrderPrecautions=false
EOF

# Create additional API settings file
cat > /home/ibkr/Jts/api.properties << EOF
# IBKR API Configuration
api.enabled=true
api.readonly=false
api.port=${IBKR_PORT}
api.trusted.ips=0.0.0.0,127.0.0.1,::1,172.17.0.0/16,192.168.0.0/16,10.0.0.0/8
api.accept.incoming=accept
api.allow.blind.trading=true
api.download.open.orders=true
api.bypass.order.precautions=false
EOF

echo "API configuration completed."
echo "Settings applied:"
echo "  - API Enabled: true"
echo "  - Read-Only: false"
echo "  - Port: ${IBKR_PORT}"
echo "  - Trusted IPs: 0.0.0.0 (all IPs allowed)"
echo "  - Allow Trading: true"
echo "  - Download Open Orders: true"

