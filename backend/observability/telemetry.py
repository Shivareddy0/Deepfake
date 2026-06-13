from prometheus_client import Counter, Histogram, Gauge, Summary
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# 1. Setup OpenTelemetry Tracing
provider = TracerProvider()
processor = BatchSpanProcessor(ConsoleSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)
tracer = trace.get_tracer("antigravity-deepfake-shield")

# 2. Setup Prometheus Metrics
LATENCY = Histogram(
    "media_processing_latency_seconds",
    "Processing latency of media assets in seconds",
    ["detector", "media_type"],
    buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
)

THROUGHPUT = Counter(
    "media_processed_total",
    "Total count of media assets analyzed",
    ["detector", "result_verdict"]
)

GPU_UTILIZATION = Gauge(
    "gpu_memory_utilization_ratio",
    "Ratio of active GPU memory consumed"
)

QUEUE_DEPTH = Gauge(
    "inference_queue_depth",
    "Depth of the dynamic batching queue"
)

C2PA_METRICS = {
    "issued": Counter("c2pa_issued_total", "Total C2PA credentials signed and embedded"),
    "verified": Counter("c2pa_verified_total", "Total C2PA verification checks executed", ["status"]),
}

ANALYST_METRICS = {
    "verdicts": Counter("analyst_verdicts_total", "Total analyst cases verified", ["action"]),
    "latencies": Summary("analyst_verdict_latency_seconds", "Time taken by analysts to submit verdict")
}

def instrument_detector(detector_name: str, media_type: str):
    """
    Decorator to trace and metricize detector calls.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(f"detect_{detector_name}") as span:
                import time
                start = time.time()
                res = func(*args, **kwargs)
                elapsed = time.time() - start
                
                # Record metrics
                LATENCY.labels(detector=detector_name, media_type=media_type).observe(elapsed)
                verdict = "fake" if res.confidence > 0.5 else "authentic"
                THROUGHPUT.labels(detector=detector_name, result_verdict=verdict).inc()
                
                # Log attributes in OpenTelemetry Span
                span.set_attribute("detector.name", detector_name)
                span.set_attribute("media.type", media_type)
                span.set_attribute("confidence", res.confidence)
                return res
        return wrapper
    return decorator
