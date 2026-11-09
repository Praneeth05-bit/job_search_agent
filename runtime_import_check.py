"""Verify that every direct SpartanScout runtime dependency imports."""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import PackageNotFoundError, version

DEPENDENCIES = {
    "fastapi": "fastapi",
    "google-genai": "google.genai",
    "httpx": "httpx",
    "langgraph": "langgraph",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "streamlit": "streamlit",
    "structlog": "structlog",
    "uvicorn": "uvicorn",
}


def main() -> None:
    """Import each dependency and report its installed version."""
    print(f"Python: {sys.version.split()[0]}")
    failures: list[str] = []

    for distribution_name, import_name in DEPENDENCIES.items():
        try:
            importlib.import_module(import_name)
            installed_version = version(distribution_name)
        except (ImportError, PackageNotFoundError) as exc:
            failures.append(f"{distribution_name}: {exc}")
            print(f"FAIL: {distribution_name} ({exc})")
        else:
            print(f"PASS: {distribution_name}=={installed_version}")

    if failures:
        details = "\n".join(failures)
        raise RuntimeError(f"Runtime import check failed:\n{details}")

    print("Runtime imports: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
