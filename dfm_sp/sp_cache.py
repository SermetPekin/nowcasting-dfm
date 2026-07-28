"""
BSD 3-Clause License

Copyright (c) 2026, Sermet Pekin (extensions and modernisation)

"""

import hashlib
import pickle
from functools import wraps
from typing import Any, Callable, Dict
from pathlib import Path
import os


class CacheHandler:
    def __init__(self, cache_dir=".func_caches", max_age_seconds=3600):
        """
        Initialize the cache handler with an optional cache directory.

        Args:
            cache_dir: Directory to store cache files (default: ".function_cache")
        """
        self.cache_dir = cache_dir

        self.max_age_seconds = max_age_seconds
        self._cache: Dict[str, Any] = {}
        self._persistent = True

        self.create_cache_dir(self.cache_dir)

    def create_cache_dir(self, effective_dir: Path):

        # Create cache directory if it doesn't exist
        if effective_dir and not os.path.exists(effective_dir):
            os.makedirs(effective_dir)

    def _make_hash(self, *args, **kwargs) -> str:
        """
        Create a hash from function arguments.
        Handles both positional and keyword arguments.
        """
        # Convert args and kwargs to a serializable format
        data = (args, sorted(kwargs.items()))
        serialized = pickle.dumps(data)
        return hashlib.sha256(serialized).hexdigest()

    def _get_cache_path(
        self, effective_cache_dir, func_name: str, hash_key: str
    ) -> str:
        """Generate a filesystem path for the cached value."""
        if not effective_cache_dir:  # self.cache_dir:
            raise ValueError("Cache directory not set for persistent caching")

        self.create_cache_dir(effective_cache_dir)  # bunu silebilirim gerek yok aslında

        return os.path.join(effective_cache_dir, f"{func_name}_{hash_key}.cache")

    def __call__(self, func: Callable) -> Callable:
        """
        Decorator that caches the function's return value based on its arguments.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):

            # Create hash key for the current arguments
            # eski ==> hash_key = self._make_hash(*args, **kwargs)
            filtered_kwargs = {k: v for k, v in kwargs.items() if k != "cache_dir"}

            # Calculate hash with the filtered kwargs
            hash_key = self._make_hash(*args, **filtered_kwargs)

            eff_verbose = kwargs.pop("verbose", False)

            # Caller can override the cache directory per-call via cache_dir kwarg.
            override_cache_dir = kwargs.pop("cache_dir", None)
            effective_cache_dir = override_cache_dir or self.cache_dir

            override_max_age_seconds = kwargs.pop("max_age_seconds", None)
            eff_max_age_seconds = override_max_age_seconds or self.max_age_seconds

            if not effective_cache_dir:
                raise ValueError("No cache_dir specified!")

            # Check in-memory cache first
            if hash_key in self._cache:
                return self._cache[hash_key]

            # If persistent, check filesystem
            if self._persistent:
                cache_path = self._get_cache_path(
                    effective_cache_dir, func.__name__, hash_key
                )

                if os.path.exists(cache_path):
                    if self._is_cache_valid(cache_path, eff_max_age_seconds):
                        if eff_verbose:
                            print("[Valid Cache]")

                        with open(cache_path, "rb") as f:
                            if eff_verbose:
                                print("<CACHE RESULT>")
                            return pickle.load(f)

            if eff_verbose:
                print("[Found No Cache. Will make a new request!]")
            # Call the actual function
            result = func(*args, **kwargs)
            # Store in memory
            self._cache[hash_key] = result

            # =============  Sonucu yazma kısmı
            # Store persistently if enabled
            if self._persistent and result is not None:

                cache_path = self._get_cache_path(
                    effective_cache_dir, func.__name__, hash_key
                )

                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)

            return result

        return wrapper

    def _is_cache_valid(self, cache_path, eff_max_age_seconds):
        """Check if the cache file exists and is not expired."""
        import time

        if not os.path.exists(cache_path):
            return False
        # Get last modification time (in seconds since epoch)
        last_modified = os.path.getmtime(cache_path)
        current_time = time.time()
        # Check if the file is older than max_age_seconds
        return (current_time - last_modified) <= eff_max_age_seconds

    def enable_persistent(self, enable: bool = True) -> None:
        """Enable or disable persistent caching to filesystem."""
        self._persistent = enable

    def clear_cache(self) -> None:
        """Clear all cached values from memory and optionally filesystem."""
        self._cache.clear()
        if self._persistent and self.cache_dir:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith(".cache"):
                    os.remove(os.path.join(self.cache_dir, filename))
