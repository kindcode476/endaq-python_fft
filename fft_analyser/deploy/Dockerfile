# Live FFT analyser.
#
#   docker build -f fft_analyser/deploy/Dockerfile -t fft-analyser .
#   docker run --rm -p 8050:8050 --env-file .env fft-analyser
#
# The container needs outbound HTTPS to the X2 API host and nothing else.
# It never listens for, or accepts, anything from the sensors.

FROM python:3.11-slim

WORKDIR /app

# build deps for scipy/numpy wheels are not needed on manylinux, but curl
# is handy for the healthcheck
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY setup.py README.md ./
COPY endaq ./endaq
COPY fft_analyser ./fft_analyser

RUN pip install --no-cache-dir -e ".[analyser]" waitress

# run as a non-root user; nothing here needs privileges
RUN useradd --create-home --shell /usr/sbin/nologin analyser \
 && chown -R analyser:analyser /app
USER analyser

EXPOSE 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8050/ >/dev/null || exit 1

# --serve runs Waitress rather than the Flask development server.
# --connect logs in at startup from the X2_* environment variables.
CMD ["python", "-m", "fft_analyser", "--serve", "--connect", \
     "--host", "0.0.0.0", "--port", "8050"]
