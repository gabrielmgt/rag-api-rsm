"""Module for metrics: metrics (e.g. request counts, latency histograms, error rates)."""

import time
from fastapi import Request
from prometheus_client import Counter, Histogram, Gauge
from starlette_prometheus import metrics

REQUEST_COUNT = Counter("http_requests_total",
                        "Total HTTP requests", 
                        ["method",
                          "endpoint", 
                          "status"])
REQUEST_DURATION = Histogram("request_duration_seconds", "Request duration in seconds")
CPU_USAGE = Gauge("cpu_usage_percent", "CPU usage percent")
MEMORY_USAGE = Gauge("memory_usage_percent", "Memory usage percent")


def setup_metrics(app):
    """Set up Prometheus metrics middleware and endpoints.

    Args:
        app: FastAPI application instance
    """
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        REQUEST_COUNT.labels(request.method, 
                             request.url.path, 
                             str(response.status_code)
                            ).inc()
        REQUEST_DURATION.observe(duration)
        return response

    # Add metrics endpoint
    app.add_route("/metrics", metrics)
