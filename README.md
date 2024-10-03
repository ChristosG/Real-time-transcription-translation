# Real-Time Transcriber and Translator Guide v2.0

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Prerequisites](#prerequisites)
  - [Install NVIDIA Drivers and Container Toolkit](#install-nvidia-drivers-and-container-toolkit)
  - [Install FFmpeg](#install-ffmpeg)
  - [Install Python and Node.js](#install-python-and-nodejs)
  - [Install Redis](#install-redis)
- [Setup](#setup)
  - [Step 1: Prepare Docker Environment](#step-1-prepare-docker-environment)
  - [Step 2: Build and Deploy Models](#step-2-build-and-deploy-models)
    - [2.a LLaMA 3.1 Engine](#2a-llama-31-engine)
    - [2.b Whisper Engine](#2b-whisper-engine)
  - [Step 3: Microservices Setup](#step-3-microservices-setup)
    - [3.a Configure Redis](#3a-configure-redis)
    - [3.b Deploy Backend Microservices](#3b-deploy-backend-microservices)
      - [FastAPI WebSocket Server](#fastapi-websocket-server)
      - [Transcription Service](#transcription-service)
      - [Translation Services](#translation-services)
- [React Web App](#react-web-app)
- [Running the Project](#running-the-project)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)
- [Common Issues I Dealt With](#common-issues-i-dealt-with)
- [Translation Services Explained](#translation-services-explained)
- [Extras: Display It on the Web!](#extras-display-it-on-the-web)
- [Contact](#contact)

## Description

Welcome to the **Real-Time Transcriber and Translator Guide**! This guide will walk you through building, deploying, and using a real-time transcription and translation system with a modern microservices architecture. I have restructured the backend into microservices, added Redis for independent transactions, and implemented experimental concepts like sliding window buffers for HTTP continuity to improve performance and scalability.

In this example, I focus on translating from German to English, but the setup can be easily adjusted for other languages supported by `whisper-large-v3`.

This entire project was built on a consumer PC with an NVIDIA 3060, showcasing that high-performance transcription and translation can be achieved without enterprise-grade hardware.

## Features

- **Microservices Architecture:** Backend services are modularized into microservices, enhancing scalability and maintainability.
- **Redis Integration:** Utilizes Redis for efficient message passing and independent transactions between services.
- **Real-Time Transcription:** Converts spoken German into text.
- **Real-Time Translation:** Translates transcribed text from German to English, including an enhanced translation for better semantics.
- **Dockerized Environment:** Ensures consistency and ease of deployment.
- **Scalable WebSockets Proxy:** Handles multiple users via WebSockets.
- **React Web App:** User-friendly interface for interaction.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Operating System:** Ubuntu
- **Docker:** Installed and running
- **NVIDIA Drivers:** Correct version installed
- **NVIDIA Container Toolkit:** Installed
- **FFmpeg:** For audio processing
- **Python 3.8+**
- **Node.js and npm:** For the React web app
- **Redis:** For message passing between microservices

### Install NVIDIA Drivers and Container Toolkit

```bash
# Update package lists
sudo apt update

# Install NVIDIA drivers (replace with the required version)
sudo apt install -y nvidia-driver-525

# Add the NVIDIA Docker repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Container Toolkit
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Restart Docker to apply changes
sudo systemctl restart docker
```

### Install FFmpeg

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### Install Python and Node.js

```bash
# Install Python 3 and pip
sudo apt install -y python3 python3-pip

# Install Node.js and npm
sudo apt install -y nodejs npm
```

### Install Redis

```bash
sudo apt update
sudo apt install -y redis-server
```

## Setup

### Step 1: Prepare Docker Environment

1. **Verify NVIDIA Drivers in Docker**

   Ensure Docker can access your NVIDIA drivers by running:

   ```bash
   docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
   ```

   You should see the output from `nvidia-smi`. If not, revisit the NVIDIA driver and container toolkit installation.

### Step 2: Build and Deploy Models

<details>
  <summary>Shout out to k2-fsa Sherpa - Triton Whisper Guide for their slick instructions on building the Whisper TRT engine.</summary>
  
  You can check out the guide [here](https://github.com/k2-fsa/sherpa/tree/master/triton/whisper).
</details>

#### 2.a LLaMA 3.1 Engine

1. **Create Engines Directory**

   ```bash
   mkdir -p engines
   cd engines
   ```

2. **Download Models**

   ```bash
   # Create assets directory
   mkdir -p assets

   # Download Whisper Large V3 model
   wget --directory-prefix=assets https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt

   # Download LLaMA 3.1 Instruct model
   wget --directory-prefix=assets https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct/resolve/main/model.bin
   ```

3. **Build Docker Image**

   ```bash
   cd build_engines_and_init_triton
   docker build -t llama_whisper .
   ```

4. **Start Builder Script**

   ```bash
   ./start_builder.sh
   ```

5. **Convert LLaMA 3.1 Checkpoint**

   Ensure all dependencies are installed. If you encounter errors related to Python libraries, install them accordingly using `pip`.

   ```bash
   docker exec -it triton-trtllm-container bash
   cd TensorRT-LLM

   python3 examples/llama/convert_checkpoint.py \
     --model_dir /engines/Meta-Llama-3.1-8B-Instruct \
     --output_dir /engines/llama31_checkpoint \
     --dtype float16 \
     --use_weight_only \
     --weight_only_precision int4
   ```

   > **Note:** The quantized model is set to `int4` for translation and short sentence semantics with `max_seq_len` set to 2000 tokens. Adjust these settings as needed by modifying `tensorrt_llm/config.pbtxt` in the `tensorrtllm_backend` directory.

6. **Build TRTLLM Engine**

   ```bash
   trtllm-build \
     --checkpoint_dir /engines/llama31_checkpoint \
     --output_dir /engines/llama31_engine \
     --gemm_plugin float16 \
     --max_batch_size 1 \
     --max_input_len 1000 \
     --max_seq_len 2000 \
     --use_paged_context_fmha enable
   ```

7. **Test the Build**

   ```bash
   tritonserver --model-repository=tensorrtllm_backend/ \
     --model-control-mode=explicit \
     --load-model=preprocessing \
     --load-model=postprocessing \
     --load-model=tensorrt_llm \
     --load-model=tensorrt_llm_bls \
     --load-model=ensemble \
     --log-verbose=2 \
     --log-info=1 \
     --log-warning=1 \
     --log-error=1 \
     --http-port 8000 \
     --grpc-port 8001 \
     --metrics-port 8002
   ```

   If you see the following logs, the server is ready for inference:

   ```
   I0919 10:12:29.013889 126 grpc_server.cc:2463] "Started GRPCInferenceService at 0.0.0.0:8001"
   I0919 10:12:29.014019 126 http_server.cc:4692] "Started HTTPService at 0.0.0.0:8000"
   I0919 10:12:29.055157 126 http_server.cc:362] "Started Metrics Service at 0.0.0.0:8002"
   ```

8. **Test Inference**

   Open another terminal and run:

   ```bash
   curl -X POST "http://localhost:8000/v2/models/ensemble/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "text_input": "Ich heiße Chris",
       "parameters": {
         "max_tokens": 100,
         "bad_words": [""],
         "stop_words": [""],
         "temperature": 0.0,
         "top_k": 50
       }
     }'
   ```

#### 2.b Whisper Engine

1. **Stop Triton Server with LLaMA**

   Press `Ctrl+C` in the terminal running the Triton server to stop it.

2. **Build Whisper Engine**

   ```bash
   docker exec -it triton-trtllm-container bash
   cd TensorRT-LLM

   # Create Whisper Checkpoint
   python3 examples/whisper/convert_checkpoint.py \
     --model_dir /engines/assets \
     --output_dir /engines/whisper_large_checkpoint \
     --model_name large-v3 \
     --use_weight_only \
     --weight_only_precision int8

   # Build TRT Encoder Engine
   trtllm-build \
     --checkpoint_dir /engines/whisper_large_checkpoint/encoder \
     --output_dir /engines/whisper_large/encoder \
     --paged_kv_cache disable \
     --moe_plugin disable \
     --enable_xqa disable \
     --max_batch_size 8 \
     --gemm_plugin disable \
     --bert_attention_plugin float16 \
     --remove_input_padding disable \
     --max_input_len 1500

   # Build TRT Decoder Engine
   trtllm-build \
     --checkpoint_dir /engines/whisper_large_checkpoint/decoder \
     --output_dir /engines/whisper_large/decoder \
     --paged_kv_cache disable \
     --moe_plugin disable \
     --enable_xqa disable \
     --max_beam_width 4 \
     --max_batch_size 8 \
     --max_seq_len 114 \
     --max_input_len 14 \
     --max_encoder_input_len 1500 \
     --gemm_plugin float16 \
     --bert_attention_plugin float16 \
     --gpt_attention_plugin float16 \
     --remove_input_padding disable
   ```

3. **Download Required Files**

   Ensure the following files are present in the `tensorrtllm_backend/whisper/1/` directory:

   ```bash
   wget --directory-prefix=tensorrtllm_backend/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken
   wget --directory-prefix=tensorrtllm_backend/whisper/1/ https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/mel_filters.npz
   ```

4. **Deploy Whisper on Triton for Testing**

   ```bash
   tritonserver --model-repository=tensorrtllm_backend/ \
     --model-control-mode=explicit \
     --load-model=whisper \
     --log-verbose=2 \
     --log-info=1 \
     --log-warning=1 \
     --log-error=1 \
     --http-port 8000 \
     --grpc-port 8001 \
     --metrics-port 8002
   ```

   If it runs without errors, press `Ctrl+C` to stop the server and exit the Docker container.

### Step 3: Microservices Setup

This section sets up the backend microservices and Redis for message passing. The microservices architecture enhances scalability and maintainability by decoupling components.

#### 3.a Configure Redis

Ensure Redis is installed:

```bash
sudo apt update
sudo apt install -y redis-server
```

Redis is used for efficient message passing between microservices. The default configuration is sufficient for this project.

#### 3.b Deploy Backend Microservices

The backend services are split into the following microservices:

- **FastAPI WebSocket Server:** Connects the web app with backend services via WebSockets, handling communication with clients.
- **Transcription Service:** Captures audio from a virtual sink, processes it using Voice Activity Detection (VAD), and transcribes it using the Whisper model via Triton Inference Server.
- **Translation Services:**
  - **Basic Translation Service:** Translates each transcription as it is received.
  - **Enhanced Translation Service:** Accumulates transcriptions in real-time using a sliding window buffer and uses the LLM to generate translations with better semantics.

##### FastAPI WebSocket Server

1. **Install Dependencies:**

   Create a virtual environment and install required packages:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install fastapi uvicorn aioredis websockets
   ```

2. **Run the FastAPI Server:**

   ```bash
   uvicorn app:app --host 0.0.0.0 --port 7000
   ```

   Ensure the server is properly connected to Redis and is listening on the appropriate WebSocket endpoint.

##### Transcription Service

1. **Install Dependencies:**

   ```bash
   pip install sounddevice webrtcvad aioredis numpy requests transformers
   ```

2. **Run the Transcription Service:**

   ```bash
   python transcription_service.py
   ```

   This service captures audio, processes it, and publishes transcriptions to Redis.

##### Translation Services

1. **Install Dependencies:**

   ```bash
   pip install aioredis requests transformers difflib langchain
   ```

2. **Run the Translation Services:**

   - **Basic Translation Service:**

     ```bash
     python translation_service.py
     ```

   - **Enhanced Translation Service:**

     ```bash
     python enhanced_translation_service.py
     ```

   Ensure they are connected to the correct Redis channels and are properly configured.

## React Web App

**Note:** This section is optional. You can adjust `App.tsx` as needed and open it directly in a browser. This approach may require modifying API endpoints.

### Prerequisites

- **Node.js and npm:** Ensure they are installed.

### Steps

1. **Install Node Modules**

   ```bash
   npm install
   ```

2. **Start the React App**

   ```bash
   npm start
   ```

## Running the Project

Assuming all steps have been completed successfully:

1. **Close All Services:** Ensure Docker containers and servers are stopped.

2. **Start Triton Inference Server:**

   ```bash
   tritonserver --model-repository=tensorrtllm_backend/ \
     --model-control-mode=explicit \
     --load-model=preprocessing \
     --load-model=postprocessing \
     --load-model=tensorrt_llm \
     --load-model=tensorrt_llm_bls \
     --load-model=ensemble \
     --http-port 8000 \
     --grpc-port 8001 \
     --metrics-port 8002
   ```

3. **Start Redis Server:**

   ```bash
   sudo service redis-server start
   ```

4. **Run the Backend Microservices:**

   - **Start the FastAPI WebSocket Server:**

     ```bash
     uvicorn app:app --host 0.0.0.0 --port 7000
     ```

   - **Start the Transcription Service:**

     ```bash
     python transcription_service.py
     ```

   - **Start the Translation Services:**

     ```bash
     python translation_service.py
     python enhanced_translation_service.py
     ```

5. **Start the React Web App:**

   ```bash
   npm start
   ```

6. **Use the Application:**

   - Access the web app in your browser.
   - Speak into your microphone or use a virtual sink to capture audio.
   - View the transcriptions and translations in real-time on the web app.

   **Tip:** For better audio capture during Zoom calls, set up a Virtual Sink environment. This captures audio directly from Zoom instead of playing it through speakers, reducing noise and delay for improved transcription quality.

## Usage

- **Real-Time Interaction:** Speak into your microphone, and the system will transcribe and translate your speech in real-time.
- **Enhanced Translations:** Benefit from improved translations with better semantics using the Enhanced Translation Service.
- **Web Interface:** Use the React web app to view transcriptions and translations.
- **Customization:** Adjust model parameters and configurations to suit different languages and use cases.
- **Scalability:** The microservices architecture and Redis integration allow for scalable deployment and efficient message passing between services.

## Dependencies

Ensure the following dependencies are installed:

- **Ubuntu Packages:**
  - `docker`
  - `nvidia-driver`
  - `nvidia-container-toolkit`
  - `ffmpeg`
  - `redis-server`
- **Python Libraries:**
  - `fastapi`
  - `uvicorn`
  - `aioredis`
  - `webrtcvad`
  - `sounddevice`
  - `numpy`
  - `requests`
  - `transformers`
  - `langchain`
  - `difflib`
- **Node.js and npm**
- **Additional Tools:**
  - `wget`
  - `curl`
  - `git`

## Troubleshooting

- **Docker NVIDIA Access Issues:**
  - Verify NVIDIA drivers and container toolkit are correctly installed.
  - Run `nvidia-smi` inside a Docker container to check access.

- **Python Dependency Errors:**
  - Ensure all required Python libraries are installed.
  - Use `pip install -r requirements.txt` to install dependencies.

- **Triton Server Errors:**
  - Check logs for specific error messages.
  - Ensure model paths and configurations are correct.

- **Redis Connection Issues:**
  - Ensure Redis server is running.
  - Verify the correct Redis host and port in your configuration.

- **React App Issues:**
  - Verify Node.js and npm are correctly installed.
  - Check API endpoint configurations in `App.tsx`.

## Common Issues I Dealt With

- **Version Mismatch Between TensorRT-LLM Engine Builder and Triton:**
  Ensure that both the TensorRT-LLM engine builder and Triton are using the exact same version to avoid compatibility issues.

- **LLaMA 3.1 Requires a Newer Transformers Library:**
  The LLaMA 3.1 model needs a more recent version of the `transformers` library than what the current TensorRT-LLM version provides. To resolve this, update the `transformers` library accordingly.

- **Docker Runtime Configuration on Ubuntu:**
  On Ubuntu, Docker may require using `--runtime=nvidia` instead of `--gpus=all` to properly utilize NVIDIA GPUs. Use the following command to verify:

  ```bash
  docker run --rm --runtime=nvidia nvidia-smi
  ```

- **Resolving Errors in the Triton-TRTLLM Container:**
  If you encounter errors within the Triton-TRTLLM container, try the following steps inside the container to uninstall conflicting packages and install the correct versions:

  ```bash
  pip3 uninstall -y tensorrt tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs tensorrt-llm torch
  pip3 install tensorrt_llm==0.13.0.dev2024082000 -U --pre --extra-index-url https://pypi.nvidia.com
  pip install -U transformers
  ```

  These commands ensure that `tensorrt_llm` and `transformers` are compatible with LLaMA 3.1.

## Translation Services Explained

### Basic Translation Service

The **Basic Translation Service** translates each transcription as it is received. It listens to the `transcriptions` channel on Redis, and for every new transcription, it uses the LLM to translate the text from German to English. The translation is then published to the `translations` channel on Redis.

**Key Features:**

- **Immediate Translation:** Translates each transcription individually without considering context.
- **Direct Publishing:** Sends translations directly to the web app via Redis.

**How It Works:**

1. **Listening to Transcriptions:**
   - Subscribes to the `transcriptions` channel in Redis.
   - Waits for new transcription messages.

2. **Translation:**
   - Uses the LLM to translate the German transcription to English.
   - Does not consider previous transcriptions or context.

3. **Publishing Translations:**
   - Publishes the translation to the `translations` Redis channel.
   - The FastAPI server broadcasts the translation to the web app via WebSockets.

### Enhanced Translation Service

The **Enhanced Translation Service** improves upon the basic translation by accumulating transcriptions in real-time and using a sliding window buffer to provide translations with better semantics. This approach allows the service to generate more coherent and context-aware translations, rather than translating individual short transcriptions that may lack complete meaning.

**Key Features:**

- **Sliding Window Buffer:** Accumulates recent transcriptions to provide context for translation.
- **Semantic Completeness Check:** Uses the LLM to determine if the accumulated text forms a semantically complete thought.
- **Similarity Checking:** Compares new translations with previous ones to avoid repeating similar translations using `difflib`.
- **Publishing Translations:** Publishes translations to Redis only when significant new information is available.

**How It Works:**

1. **Listening to Transcriptions:**
   - Subscribes to the `transcriptions` channel in Redis.
   - Waits for new transcription messages.

2. **Accumulating Transcriptions:**
   - Incoming transcriptions are added to a buffer.
   - The buffer has a maximum size (`MAX_BUFFER_SIZE`) to limit the amount of accumulated text.

3. **Semantic Analysis:**
   - Checks if the accumulated text is semantically complete by querying the LLM.
   - If the text forms a complete thought or if the buffer is full, it proceeds to translation.

4. **Translation:**
   - Uses the LLM to translate the accumulated German text into English.
   - Ensures that translations are coherent and contextually relevant.

5. **Similarity Filtering:**
   - Compares the new translation with previous ones to avoid redundancy.
   - Only publishes the translation if it differs significantly from recent translations.

6. **Publishing Translations:**
   - Publishes the translation to the `better_translations` Redis channel.
   - The FastAPI server broadcasts the translation to the web app via WebSockets.

## Extras: Display It on the Web!

To make the application accessible via the web so that my coworkers can access it, I set up Nginx for HTTPS encryption and configured my PC to act as a web server.

1. **Install Nginx:**

   ```bash
   sudo apt update
   sudo apt install -y nginx
   ```

2. **Create Nginx Configuration File:**

   Create a new Nginx configuration file, e.g., `/etc/nginx/sites-available/chris`:

   ```bash
   sudo nano /etc/nginx/sites-available/chris
   ```

   Add the following configuration (replace `your_dns_name` with your actual DNS name):

   ```nginx
   server {
       listen 80;
       server_name your_dns_name.duckdns.org;

       # Redirect all HTTP requests to HTTPS
       return 301 https://$host$request_uri;
   }

   server {
       listen 443 ssl;
       server_name your_dns_name.duckdns.org;

       ssl_certificate /etc/letsencrypt/live/your_dns_name.duckdns.org/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/your_dns_name.duckdns.org/privkey.pem;
       include /etc/letsencrypt/options-ssl-nginx.conf;
       ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

       # Security Enhancements
       add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
       add_header X-Frame-Options DENY;
       add_header X-Content-Type-Options nosniff;

       # Proxy WebSocket connections to FastAPI
       location /ws {
           proxy_pass http://localhost:7000/ws;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "Upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_read_timeout 86400;
           proxy_send_timeout 86400;
       }

       # Proxy to React development server
       location / {
           proxy_pass http://localhost:3000/;
           proxy_http_version 1.1;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
           proxy_set_header Connection "";
           proxy_buffering off;
       }
   }
   ```

3. **Enable the Nginx Configuration:**

   ```bash
   sudo ln -s /etc/nginx/sites-available/chris /etc/nginx/sites-enabled/
   ```

4. **Obtain SSL Certificates:**

   Use Let's Encrypt to obtain free SSL certificates for HTTPS:

   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d your_dns_name.duckdns.org
   ```

5. **Reload Nginx:**

   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

6. **Port Forwarding on Your Router:**

   - **Access Router Settings:** Log in to your router's administration page.
   - **Forward Necessary Ports:** Forward the following ports to your local machine:
     - **Port 80:** For HTTP traffic.
     - **Port 443:** For HTTPS traffic.
     - **Port 7000:** For your FastAPI WebSocket server.

7. **Access the Application:**

   - After setting up, my coworkers can access the web application using `https://your_dns_name.duckdns.org`.
   - Ensure that firewall rules allow incoming connections on the necessary ports.

   **Note:** In the Nginx configuration, I have set the `proxy_pass` for the React app to `http://localhost:3000` because I'm exposing the development server directly. I could build the React app for production and serve it with Nginx, but I chose to expose the dev server for simplicity.

This setup allows me to expose my PC as a web server securely over HTTPS, making it accessible to my coworkers.

## Contact

- **Author:** Christos Grigoriadis
- **Email:** cgrigoriadis@outlook.com
```
