from pathlib import Path

import pytest

pytest.importorskip("yaml")

from hippocampal_memory.migration import VaultMigrator


def test_vault_migration_is_staged_and_resolves_structure(tmp_path: Path) -> None:
    source = tmp_path / "Knowledge"
    source.mkdir()
    (source / "First.md").write_text(
        "# Same durable idea\n\nSee [[Missing Source]].", encoding="utf-8"
    )
    (source / "Second.md").write_text(
        "# Same durable idea\n\nSee [[Missing Source]].", encoding="utf-8"
    )
    staging = tmp_path / "Knowledge.staging"
    report = VaultMigrator(source).stage(staging)
    assert source.exists()
    assert report.before.valid is False
    assert report.duplicates_archived == 1
    assert report.link_stubs_created == 1
    assert report.after.valid is True
    assert (staging / "Home.md").exists()
    assert (staging / "Maps" / "Finance.md").exists()
    assert list((staging / "Archive" / "Duplicates").glob("*.md"))


def test_cutover_rejects_unvalidated_staging(tmp_path: Path) -> None:
    source = tmp_path / "Knowledge"
    staging = tmp_path / "Knowledge.staging"
    source.mkdir()
    staging.mkdir()
    (source / "A.md").write_text("# A", encoding="utf-8")
    (staging / "bad.md").write_text("# no frontmatter", encoding="utf-8")
    with pytest.raises(RuntimeError):
        VaultMigrator(source).cutover(staging, tmp_path / "Knowledge.archive")
