"""Module for metrics: metrics (e.g. request counts, latency histograms, error rates)."""

from prometheus_client import Counter, Histogram, Gauge
from starlette_prometheus import metrics, PrometheusMiddleware

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
    # Add Prometheus middleware
    app.add_middleware(PrometheusMiddleware)

    # Add metrics endpoint
    app.add_route("/metrics", metrics)
