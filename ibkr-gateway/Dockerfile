# IBKR Gateway Docker Container - Secure Implementation
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required packages
RUN apt-get update && apt-get install -y \
    openjdk-11-jdk \
    xvfb \
    x11vnc \
    fluxbox \
    wget \
    curl \
    unzip \
    python3 \
    python3-pip \
    socat \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# Create ibkr user
RUN useradd -m -s /bin/bash ibkr

# Create necessary directories
RUN mkdir -p /opt/ibkr /home/ibkr/.vnc /home/ibkr/Jts /home/ibkr/ibc

# Copy the IBKR Gateway installer
COPY ibgateway-latest-standalone-linux-x64.sh /opt/ibkr/

# Make installer executable
RUN chmod +x /opt/ibkr/ibgateway-latest-standalone-linux-x64.sh

# Install IBKR Gateway silently
RUN cd /opt/ibkr && \
    ./ibgateway-latest-standalone-linux-x64.sh -q -dir /opt/ibkr/IBGateway

# Download and install IBC from official source
RUN cd /tmp && \
    wget https://github.com/IbcAlpha/IBC/releases/download/3.22.0/IBCLinux-3.22.0.zip && \
    unzip IBCLinux-3.22.0.zip -d /home/ibkr/ibc && \
    chmod +x /home/ibkr/ibc/*.sh && \
    rm IBCLinux-3.22.0.zip

# Set ownership
RUN chown -R ibkr:ibkr /opt/ibkr /home/ibkr

# Copy our custom configuration files
COPY config/ /home/ibkr/Jts/
COPY scripts/ /opt/ibkr/scripts/
COPY ibc-config/ /home/ibkr/ibc/

# Make scripts executable
RUN chmod +x /opt/ibkr/scripts/*.sh

# Set ownership for config files
RUN chown -R ibkr:ibkr /home/ibkr/Jts/ /opt/ibkr/scripts/ /home/ibkr/ibc/

# Expose ports
# 4001 - Gateway API port for live trading  
# 4002 - Gateway API port for paper trading
# 5900 - VNC port
EXPOSE 4001 4002 5900

# Switch to ibkr user
USER ibkr

# Set working directory
WORKDIR /home/ibkr

# Environment variables
ENV DISPLAY=:1
ENV IBKR_HOST=127.0.0.1
ENV IBKR_PORT=4002
ENV IBKR_CLIENT_ID=1
ENV ACCOUNT_CODE=DU2838017
ENV IBKR_USERNAME=isht1430
ENV IBKR_PASSWORD=Gamma@1430Nav9464

# Start script
CMD ["/opt/ibkr/scripts/start-with-ibc.sh"]

