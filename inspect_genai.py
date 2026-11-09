"""One-off, read-only check of the installed google-genai Interactions API.

Makes NO network calls and needs NO API key. Prints the exact method
signature, response fields, usage fields, and exception classes of the SDK
version installed in this project, so the Gemini adapter matches it exactly.

Run from the project root:
    uv run python inspect_genai.py
"""

import importlib
import inspect
import typing
from importlib.metadata import version


def show_model(cls: object, indent: str = "  ", depth: int = 0) -> None:
    """Print a pydantic model's fields and, one level down, nested model fields."""
    fields = getattr(cls, "model_fields", None)
    if not fields:
        print(f"{indent}(not a pydantic model: {cls!r})")
        return
    for name, info in fields.items():
        print(f"{indent}{name}: {info.annotation}")
        if depth == 0 and name in {
            "usage",
            "usage_metadata",
            "outputs",
            "output",
            "steps",
        }:
            for arg in typing.get_args(info.annotation) or (info.annotation,):
                inner_args = typing.get_args(arg) or (arg,)
                for inner in inner_args:
                    if getattr(inner, "model_fields", None):
                        print(f"{indent}  -> {getattr(inner, '__name__', inner)}:")
                        show_model(inner, indent + "      ", depth + 1)


def main() -> None:
    from google import genai

    print("google-genai version:", version("google-genai"))
    client = genai.Client(api_key="not-a-real-key-no-calls-are-made")
    create = client.interactions.create

    print("\n== interactions.create signature ==")
    try:
        print(inspect.signature(create))
    except (TypeError, ValueError) as exc:
        print("could not read signature:", exc)

    print("\n== return type and its fields ==")
    try:
        hints = typing.get_type_hints(create)
        returned = hints.get("return")
        print("return:", returned)
        candidates = typing.get_args(returned) or (returned,)
        for cls in candidates:
            if getattr(cls, "model_fields", None):
                print(f"\n{cls.__module__}.{cls.__name__}:")
                show_model(cls)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print("could not resolve return type:", repr(exc))

    print("\n== exception classes ==")
    module_name = type(client.interactions).__module__
    print("interactions client module:", module_name)
    package = module_name
    while package:
        for suffix in ("_exceptions", "errors"):
            try:
                mod = importlib.import_module(f"{package}.{suffix}")
            except ImportError:
                continue
            names = [
                n
                for n, obj in vars(mod).items()
                if inspect.isclass(obj)
                and issubclass(obj, BaseException)
                and obj.__module__ == mod.__name__
            ]
            if names:
                print(f"{mod.__name__}: {sorted(names)}")
        package = package.rpartition(".")[0]

    print("\n== HttpOptions fields (client-level timeout/retry) ==")
    try:
        from google.genai import types

        show_model(types.HttpOptions)
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        print("could not read HttpOptions:", repr(exc))


if __name__ == "__main__":
    main()
