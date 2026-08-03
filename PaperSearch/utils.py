import re


def sanitize_filename(name: str) -> str:
    """Turn a DOI (or any string) into a filesystem-safe id.

    Used consistently across `download_manager.py` (PDF filename) and
    `convert_mineru.py` (MinerU output subfolder name) so both stages agree
    on the same `paper_id` for a given DOI.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)