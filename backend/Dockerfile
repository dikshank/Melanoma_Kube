# ============================================================
# Backend Dockerfile
# Phase 4 | FastAPI melanoma inference server
# ============================================================

FROM python:3.10.21-slim-bookworm

# ── System dependencies ───────────────────────────────────────────────────────
# RUN apt-get update && apt-get install -y \
#     gcc \
#     git \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
# We will copy the backend source code into this directory in the next step because it is the working directory for the backend service and it is where the backend service will be run from. This is important because the backend service will need to access the source code in order to run.
WORKDIR /app      

# ── Install Python dependencies ───────────────────────────────────────────────
# This copies the requirements.txt file from the backend directory into the working directory of the Docker image. This is important because the requirements.txt file contains a list of all the Python dependencies that are required to run the backend service. By copying this file into the working directory, we can then use it to install all the necessary dependencies in the next step.
COPY requirements.txt .   

# Since I don't have a GPU, why bother installing the whole torch with CUDA overheads
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --trusted-host download.pytorch.org \
    --trusted-host download-r2.pytorch.org \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org

# Now that we have copied the requirements.txt file into the working directory, we can use it to install all the necessary Python dependencies for the backend service. The --no-cache-dir option is used to prevent pip from caching the installed packages, which helps to reduce the size of the Docker image.
RUN pip install --no-cache-dir -r requirements.txt 

# ── Copy backend source ───────────────────────────────────────────────────────
# This copies the entire backend source code from the backend directory into the working directory of the Docker image. This is important because the backend service will need access to all of its source code in order to run properly.
COPY . .  

# ── Create logs directory ─────────────────────────────────────────────────────
# This creates a directory called logs in the working directory of the Docker image. This is important because the backend service will need a place to store its log files, and this directory will be used for that purpose.
RUN mkdir -p logs  

# ── Expose port ───────────────────────────────────────────────────────────────
# This is decorative and is used to indicate that the backend service will be listening on port 8000. This is important because it allows other services to know which port to use when communicating with the backend service.
EXPOSE 8000 

# ── Start FastAPI with uvicorn ────────────────────────────────────────────────
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# Reload is not needed in kubernetes because the container will be restarted if it crashes, so we don't need to use the --reload option. The --host option is set to 0.0.0.0 to make the server accessible from outside the container.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 