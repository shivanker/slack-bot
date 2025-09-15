FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Install system dependencies (combine into one RUN command to reduce layers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git ffmpeg libsm6 libxext6 xvfb xauth x11-utils \
    build-essential python3-dev vim \
    g++ make cmake unzip libcurl4-openssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv tool
RUN pip install pip -U
RUN pip install uv

# Copy project build files
COPY owl/pyproject.toml .
COPY owl/README.md .

# Create virtual environment and install dependencies
RUN uv venv .venv --python=3.12 && \
    . .venv/bin/activate && \
    uv pip install -e .

# Copy project runtime files
COPY owl/owl/ ./owl/
COPY owl/licenses/ ./licenses/
COPY owl/assets/ ./assets/
COPY owl/examples/ ./examples/

# Create startup script
RUN printf '#!/bin/bash\nxvfb-run --auto-servernum --server-args="-screen 0 1280x960x24" python "$@"' > /usr/local/bin/xvfb-python && \
    chmod +x /usr/local/bin/xvfb-python

# Create welcome script
RUN printf '#!/bin/bash\necho "Welcome to the OWL Project Docker environment!"\necho "Welcome to OWL Project Docker environment!"\necho ""\necho "Available scripts:"\nls -1 *.py | grep -v "__" | sed "s/^/- /"\necho ""\necho "Run examples:"\necho "  xvfb-python run.py                     # Run default script"\necho "  xvfb-python run_deepseek_example.py      # Run DeepSeek example"\necho ""\necho "Or use custom query:"\necho "  xvfb-python run.py \"Your question\""\necho ""' > /usr/local/bin/owl-welcome && \
    chmod +x /usr/local/bin/owl-welcome

# Set working directory
WORKDIR /app/owl

# Camel Owl startup command
# CMD ["/bin/bash", "-c", "owl-welcome && /bin/bash"]

# Include global arg in this stage of the build
ARG FUNCTION_DIR="/var/task"
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY requirements.txt ${FUNCTION_DIR}
COPY *.py ${FUNCTION_DIR}

# Install the function's dependencies
RUN pip install --target ${FUNCTION_DIR} awslambdaric

# Install the specified packages
RUN pip install -r requirements.txt

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]