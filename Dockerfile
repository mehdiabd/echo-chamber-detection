# Base image comes from the internal Docker Hub proxy. A bare `python:3.11-slim-bookworm`
# pulls from public Docker Hub, which the build runners cannot reach (403).
FROM docker.repo.stinascloud.ir/library/python:3.11-slim-bookworm

WORKDIR /app

# Point apt at the internal Debian mirror before the first update. Bookworm keeps its
# sources in the deb822 file /etc/apt/sources.list.d/debian.sources; the classic
# /etc/apt/sources.list is handled too so this keeps working if the base image changes.
RUN for f in /etc/apt/sources.list.d/debian.sources /etc/apt/sources.list; do \
        [ -f "$f" ] && sed -i \
            -e 's|http://deb.debian.org/debian|https://nexus.repo.stinascloud.ir/repository/debian|g' \
            -e 's|http://security.debian.org/debian-security|https://nexus.repo.stinascloud.ir/repository/debian|g' \
            -e 's|http://deb.debian.org/debian-security|https://nexus.repo.stinascloud.ir/repository/debian|g' \
            "$f" || true; \
    done && \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# pip must also go through Nexus — pypi.org is unreachable from the runners.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        --index-url https://nexus.repo.stinascloud.ir/repository/pypi/simple

COPY . .

ENV MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

EXPOSE 8765

CMD ["python", "-B", "echo_chamber_api.py", "--host", "0.0.0.0"]
