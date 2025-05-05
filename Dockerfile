FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up timezone to prevent interactive prompts
RUN apt-get update && apt-get install -y --no-install-recommends tzdata

# Install GStreamer and its plugins
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    python3-gst-1.0 \
    python3-numpy \
    pkg-config \
    libcairo2-dev \
    libgirepository1.0-dev \
    gir1.2-gtk-3.0 \
    ffmpeg \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Make system site-packages accessible to the virtual environment
RUN echo "/usr/lib/python3/dist-packages" > /opt/venv/lib/python3.10/site-packages/system-site-packages.pth

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create app directory
WORKDIR /app

# Copy application code
COPY main.py .
COPY api.py .
COPY start.sh .

# Make the startup script executable
RUN chmod +x start.sh

# Create config directory if it doesn't exist
RUN mkdir -p /app/config

# Create volume for environment file
VOLUME /app/config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app:/usr/lib/python3/dist-packages
ENV GST_DEBUG=2
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV MONGODB_URI=mongodb://admin:password@mongo:27017/
ENV MONGODB_DB_NAME=bark_detector

# Expose API port
EXPOSE 3000

# Run the application using the startup script
CMD ["./start.sh"] 