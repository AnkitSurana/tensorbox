FROM python:3.11-slim

# Install system dependencies, databases & development tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    procps \
    net-tools \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    graphviz \
    postgresql \
    postgresql-contrib \
    redis-server \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install code-server (VS Code in the Browser)
RUN curl -fsSL https://raw.githubusercontent.com/coder/code-server/main/install.sh | sh

WORKDIR /workspace

# Upgrade pip & build tools
RUN pip install --no-cache-dir --upgrade pip wheel

# Install Python requirements (deterministic lockfile)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy NLP models
RUN pip install --no-cache-dir https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.7.1/en_core_web_md-3.7.1-py3-none-any.whl || \
    (python -m spacy download en_core_web_sm || true)

# Bake tensorbox utilities, curriculum notebooks, projects, tests, and data into /opt/tensorbox
COPY utils /opt/tensorbox/utils
COPY tests /opt/tensorbox/tests
COPY notebooks /opt/tensorbox/notebooks
COPY projects /opt/tensorbox/projects
COPY data /opt/tensorbox/data
COPY .env.example /opt/tensorbox/.env.example
RUN ln -s /opt/tensorbox /opt/devbox
ENV PYTHONPATH="/workspace:/opt/tensorbox:/opt/devbox"

# Install entrypoint & CLI utilities (tensorbox & devbox alias)
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh && \
    ln -s /usr/local/bin/entrypoint.sh /usr/local/bin/tensorbox && \
    ln -s /usr/local/bin/entrypoint.sh /usr/local/bin/devbox

# Create standard workspace folders
RUN mkdir -p /workspace/projects /workspace/notebooks /workspace/data /workspace/models

# Expose ports:
# 8080: VS Code (code-server)
# 8888: JupyterLab
# 8501: Streamlit
# 5000: MLflow / Web API
# 6006: TensorBoard
EXPOSE 8080 8888 8501 5000 6006

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

