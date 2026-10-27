#!/usr/bin/env python3
"""Run the FastAPI anomaly detection server."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Start the Industrial Anomaly Detection API service.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development.")
    args = parser.parse_args()

    print(f"Starting Industrial Vision Anomaly Detection API on http://{args.host}:{args.port}")
    uvicorn.run("iad.api:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
