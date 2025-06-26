# IBKR Gateway Docker Container

This Docker container provides a fully configured Interactive Brokers Gateway with API access enabled for algorithmic trading.

## Features

- **Full API Access**: Read-write API enabled (not read-only)
- **Trading Enabled**: Allows placing orders and managing positions
- **Network Access**: Configured to accept connections from any IP
- **VNC Access**: Remote GUI access for manual configuration if needed
- **Persistent Data**: Configuration and logs are persisted across container restarts

## Configuration

The container is configured with the following settings:

### API Settings
- **API Enabled**: Yes
- **Read-Only Mode**: Disabled (full trading access)
- **Trusted IPs**: All IPs allowed (0.0.0.0/0)
- **Port**: 7497 (configurable via environment variables)
- **Accept Incoming Connections**: Automatic
- **Allow Blind Trading**: Enabled
- **Download Open Orders**: Enabled

### Connection Settings
- **Host**: 127.0.0.1 (localhost)
- **Port**: 4002 (Gateway Paper) / 4001 (Gateway Live) / 7497 (TWS)
- **Client ID**: 1
- **Account**: DU2838017 (paper trading account)

## Quick Start

1. **Build the container:**
   ```bash
   docker compose build
   ```

2. **Start the container:**
   ```bash
   docker compose up -d
   ```

3. **Complete initial login via VNC:**
   - Connect to `localhost:5900` with a VNC client
   - Login with provided credentials
   - Enable API access in Gateway settings

4. **Test the connection:**
   ```bash
   python3 test_connection.py
   ```

## Environment Variables

You can customize the configuration by modifying the `.env` file:

```bash
# IBKR Connection Settings
IBKR_HOST=127.0.0.1
IBKR_PORT=4002          # 4002 for paper, 4001 for live Gateway
IBKR_CLIENT_ID=1

# Account Settings  
ACCOUNT_CODE=DU2838017  # Your paper or live account number

# Login Credentials
IBKR_USERNAME=isht1430
IBKR_PASSWORD=Gamma@1430Nav9464
```

## Port Mappings

- **4001**: Gateway API port (live trading)
- **4002**: Gateway API port (paper trading) - **DEFAULT**
- **5900**: VNC port for remote GUI access

**Note**: Port 7497 is for TWS (Trader Workstation), not Gateway.

## Testing the Connection

After starting the container, you can test the API connection using Python:

```python
from ib_insync import *

# Connect to the gateway
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

# Test connection
print("Connected:", ib.isConnected())
print("Account:", ib.managedAccounts())

# Disconnect
ib.disconnect()
```

## Important Notes

1. **Paper Trading**: The default configuration uses paper trading mode
2. **Live Trading**: To switch to live trading, change the port to 4001 and update trading mode
3. **Authentication**: You'll need to log in through the VNC interface on first run
4. **Firewall**: Ensure the required ports are open in your firewall
5. **Security**: In production, consider restricting trusted IPs for security

## Troubleshooting

1. **Connection Refused**: Check if the container is running and ports are mapped correctly
2. **Authentication Required**: Connect via VNC to complete the login process
3. **API Not Enabled**: The container automatically configures API settings, but manual verification via VNC may be needed
4. **Port Conflicts**: Ensure the ports are not already in use by other applications

## File Structure

```
ibkr-gateway-docker/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker Compose configuration
├── .env                    # Environment variables
├── config/                 # Configuration files
│   ├── jts.ini            # Main IBKR configuration
│   └── ibgateway.vmoptions # JVM options
├── scripts/               # Startup scripts
│   ├── start.sh          # Main startup script
│   └── configure-api.sh  # API configuration script
└── README.md             # This file
```

## Support

For issues related to:
- **IBKR Gateway**: Check Interactive Brokers documentation
- **API Usage**: Refer to IB API documentation
- **Container Issues**: Check Docker logs and this README

