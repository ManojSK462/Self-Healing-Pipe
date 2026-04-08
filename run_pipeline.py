#!/usr/bin/env python3
"""
Entry point for the self-healing ML pipeline.

Usage:
  python run_pipeline.py                    # full pipeline: train + monitor + serve
  python run_pipeline.py --bootstrap-only   # just train and promote, no serving
  python run_pipeline.py --serve            # start serving (assumes model exists)
  python run_pipeline.py --drift-check      # run a single drift check
"""

import argparse
import logging
import sys
import threading
import signal
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# quiet down noisy libraries
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("feast").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)

logger = logging.getLogger("selfhealing")


def start_api_server():
    """Launch the FastAPI server in a background thread."""
    import uvicorn
    from config.settings import SERVING_HOST, SERVING_PORT

    logger.info(f"Starting API server on {SERVING_HOST}:{SERVING_PORT}")
    uvicorn.run(
        "src.serving.app:app",
        host=SERVING_HOST,
        port=SERVING_PORT,
        log_level="warning",
        access_log=False,
    )


def run_full_pipeline(args):
    """Bootstrap → serve → monitor."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()

    # graceful shutdown
    def handle_signal(sig, frame):
        logger.info("Shutting down...")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # step 1: bootstrap
    version = orchestrator.bootstrap()
    logger.info(f"Pipeline bootstrapped with model {version}")

    if args.bootstrap_only:
        logger.info("Bootstrap-only mode — exiting.")
        return

    # step 2: start API server in background
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    time.sleep(2)  # let the server bind

    # step 3: drift monitoring loop
    logger.info(f"Starting drift monitor (interval={args.monitor_interval}s, "
                f"max_iterations={args.max_iterations})")
    orchestrator.run_drift_monitor(
        interval_seconds=args.monitor_interval,
        max_iterations=args.max_iterations,
    )


def run_serve_only():
    """Just start the API server (model must already exist)."""
    start_api_server()


def run_drift_check():
    """Single drift check — useful for cron-style invocations."""
    from src.pipeline.orchestrator import PipelineOrchestrator

    orchestrator = PipelineOrchestrator()
    orchestrator.bootstrap()
    result = orchestrator.check_and_heal()

    logger.info(f"Drift check complete. Retrained: {result['retrained']}, "
                f"Promoted: {result['promoted']}")
    if result["drift_report"]:
        logger.info(result["drift_report"].summary())


def main():
    parser = argparse.ArgumentParser(
        description="Self-Healing ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bootstrap-only", action="store_true",
        help="Run bootstrap (train + promote) then exit",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Start serving only (model must already exist)",
    )
    parser.add_argument(
        "--drift-check", action="store_true",
        help="Run a single drift check cycle",
    )
    parser.add_argument(
        "--monitor-interval", type=int, default=30,
        help="Seconds between drift checks (default: 30)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=0,
        help="Max drift check iterations (0=unlimited, default: 0)",
    )

    args = parser.parse_args()

    if args.serve:
        run_serve_only()
    elif args.drift_check:
        run_drift_check()
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
