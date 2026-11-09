"""Load versioned prompt files that ship inside this package.

Prompts live next to this module as ``<name>_v<version>.md``. Each loaded
prompt carries an id such as ``search_agent@v1`` so traces and evaluation
reports record exactly which prompt produced a result.
"""

from functools import lru_cache
from importlib.resources import files

from pydantic import BaseModel, ConfigDict, Field

_PACKAGE = "job_search_agent.prompts"


class Prompt(BaseModel):
    """A prompt's text together with the name and version it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    version: int = Field(ge=1)
    text: str = Field(min_length=1)

    @property
    def id(self) -> str:
        """Stable identifier for traces and evals, such as ``search_agent@v1``."""
        return f"{self.name}@v{self.version}"


@lru_cache
def load_prompt(name: str, version: int) -> Prompt:
    """Read ``<name>_v<version>.md`` from the prompts package.

    Results are cached, so each file is read from disk only once.

    Raises:
        FileNotFoundError: If no prompt file exists for that name and version.
    """
    filename = f"{name}_v{version}.md"
    resource = files(_PACKAGE) / filename
    if not resource.is_file():
        raise FileNotFoundError(f"Prompt not found: {filename} in {_PACKAGE}")
    text = resource.read_text(encoding="utf-8").strip()
    return Prompt(name=name, version=version, text=text)
