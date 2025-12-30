# gunicorn.conf.py - Configuration for Render deployment
# This ensures all logs are visible in Render terminal

import sys
import os

# Bind to Render's PORT
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"

# Single worker (free tier)
workers = 1
worker_class = "sync"
threads = 1

# Timeouts
timeout = 120
graceful_timeout = 30
keepalive = 5

# Logging - CRITICAL for visibility
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# FORCE UNBUFFERED OUTPUT
raw_env = [
    'PYTHONUNBUFFERED=1'
]

# Pre-fork hooks for diagnostics
def on_starting(server):
    print("=" * 80, flush=True)
    print("GUNICORN: Starting server...", flush=True)
    print(f"Workers: {workers}", flush=True)
    print(f"Bind: {bind}", flush=True)
    print(f"Timeout: {timeout}s", flush=True)
    print("=" * 80, flush=True)
    sys.stdout.flush()

def when_ready(server):
    print("\n" + "=" * 80, flush=True)
    print("GUNICORN: Server is ready!", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()

def on_exit(server):
    print("\n" + "=" * 80, flush=True)
    print("GUNICORN: Server shutting down...", flush=True)
    print("=" * 80 + "\n", flush=True)
    sys.stdout.flush()

# Worker lifecycle
def post_worker_init(worker):
    print(f"\nWORKER {worker.pid}: Initialized", flush=True)
    sys.stdout.flush()
