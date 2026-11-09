from pathlib import Path

from job_search_agent.domain.models import Opportunity


def load_opportunity(path: Path) -> Opportunity:
    """Load and validate one opportunity from a UTF-8 JSON file."""
    json_text = path.read_text(encoding="utf-8")
    return Opportunity.model_validate_json(json_text)


def load_opportunities(directory: Path) -> list[Opportunity]:
    """Load every opportunity JSON file in a folder, sorted by id.

    Raises:
        FileNotFoundError: If the folder does not exist. Failing loudly here
            prevents a wrong path from silently producing zero results.
        ValueError: If two files share the same opportunity id.
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Opportunity folder not found: {directory}")

    records: list[Opportunity] = []
    seen_ids: set[str] = set()

    for path in directory.glob("*.json"):
        record = load_opportunity(path)

        if record.id in seen_ids:
            raise ValueError(f"Duplicate opportunity id: {record.id} in {path.name}")

        seen_ids.add(record.id)
        records.append(record)

    return sorted(records, key=lambda r: r.id)
