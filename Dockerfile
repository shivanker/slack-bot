FROM amazonlinux:2 AS build

# Install required packages
RUN yum update -y && \
    yum install -y libSM libXext xvfb && \
    yum clean all

FROM public.ecr.aws/lambda/python:3.12

# Copy requirements.txt
COPY ../requirements.txt ${LAMBDA_TASK_ROOT}

RUN pip install pip -U
# Install the specified packages
RUN pip install -r requirements.txt

# Copy all code
COPY ../*.py ${LAMBDA_TASK_ROOT}

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    # PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/app/.venv/bin:$PATH"

COPY --from=build /usr/lib64/libSM.so.6 /usr/lib64/
COPY --from=build /usr/lib64/libXext.so.6 /usr/lib64/


RUN dnf update -y
RUN dnf install -y git tar gcc gcc-c++ make wget xz mesa-libGL jq unzip
RUN dnf install -y vim
    # build-essential curl python3-dev \ TODO
RUN dnf install -y libX11 libXcomposite libXcursor libXdamage libXext libXi \
    libXtst cups-libs libXScrnSaver libXrandr alsa-lib pango \
    atk at-spi2-atk gtk3 libdrm mesa-libgbm \
    xorg-x11-server-Xvfb xorg-x11-xauth dbus-glib nss

# Install ffmpeg and ffprobe for ARM64
ADD https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-arm64-static.tar.xz .
RUN tar xvf ffmpeg-release-arm64-static.tar.xz && \
    mv ffmpeg-*-arm64-static/ffmpeg ffmpeg-*-arm64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-release-arm64-static*

# Set working directory
WORKDIR /app

# Install uv tool
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

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
WORKDIR ${LAMBDA_TASK_ROOT}
CMD [ "lambda_function.handler" ]