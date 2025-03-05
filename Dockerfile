FROM python:3.12.3

# Set environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    netcat-openbsd \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Foundry for Anvil
RUN curl -L https://foundry.paradigm.xyz | bash
RUN /root/.foundry/bin/foundryup

# Add Foundry to PATH
ENV PATH="/root/.foundry/bin:${PATH}"

# Install Poetry
RUN pip install poetry

# Set working directory
WORKDIR /app

# Copy only requirements to cache them in docker layer
COPY pyproject.toml poetry.lock* /app/

# Configure poetry to not use virtualenvs inside Docker
RUN poetry config virtualenvs.create false

# Install dependencies - specify CPU-only torch
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Now install the rest of the dependencies
RUN poetry install --no-interaction --no-ansi --no-root
RUN poetry install --extras server --no-root
RUN poetry install --extras rag --no-root
RUN poetry install --extras agents --no-root

# Copy project
COPY . /app/

# Create an entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Run the application
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--server", "--host", "0.0.0.0", "--port", "8000"]
