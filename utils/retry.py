import time
import functools

def retry_with_backoff(retries=3, backoff_factor=5):
    """Retry decorator with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == retries:
                        raise
                    wait = backoff_factor * (2 ** (attempt - 1))
                    print(f"Retry {attempt}/{retries} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
        return wrapper
    return decorator