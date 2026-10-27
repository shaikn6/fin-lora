"""Every Hugging Face repository the project names is pinned to a full commit hash."""
import re
from pathlib import Path

from finlora.pins import REVISIONS, revision

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [*ROOT.glob("*.py"), *(ROOT / "finlora").glob("*.py")]
# "owner/name" string literals handed to a download call, directly or through a module constant.
REPO_ID = re.compile(r'"([A-Za-z0-9][\w.-]*/[\w.-]+)"')
DOWNLOAD = re.compile(r"from_pretrained\(|load_dataset\(|hf_hub_download\(")
ID_CONSTANT = re.compile(r'^(?:MODEL|BASE|[A-Z_]+_ID)\s*=\s*"([^"]+)"', re.M)


def named_repositories():
    found = set()
    for path in SOURCES:
        text = path.read_text()
        found.update(ID_CONSTANT.findall(text))
        for line in text.splitlines():
            if DOWNLOAD.search(line):
                found.update(REPO_ID.findall(line))
    return {r for r in found if not r.startswith(("exports/", "adapters/", "data/"))}


def test_every_named_repository_is_pinned():
    named = named_repositories()
    assert named, "the scan found no repositories; the patterns no longer match the code"
    assert named <= set(REVISIONS), f"unpinned: {sorted(named - set(REVISIONS))}"


def test_pins_are_full_commit_hashes():
    assert all(re.fullmatch(r"[0-9a-f]{40}", rev) for rev in REVISIONS.values())


def test_download_calls_pass_a_revision():
    missing = [
        f"{path.name}:{n}"
        for path in SOURCES
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if re.search(r"(?:AutoTokenizer|AutoModelFor\w+)\.from_pretrained\(|load_dataset\(|hf_hub_download\(", line)
        and "revision=" not in line
    ]
    assert not missing, f"download without a revision: {missing}"


def test_unlisted_or_local_ids_fall_back_to_the_default():
    assert revision("adapters/lora_news_0.5b") is None
