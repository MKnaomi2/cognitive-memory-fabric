"""Transactional Obsidian vault repair and concept-centric migration."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

_LINK = re.compile(r"(?<!!)\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_FULL_LINK = re.compile(
    r"(?<!!)\[\[([^\]|#]+)(#[^\]|]+)?(?:\|([^\]]+))?\]\]"
)
_NAMESPACE = uuid.UUID("5683ce1a-b59b-40c8-a908-1dcf69aa73e7")
_REQUIRED = {
    "id",
    "title",
    "aliases",
    "type",
    "domain",
    "status",
    "created",
    "updated",
    "source_type",
    "source_ref",
    "sensitivity",
    "review_on",
    "tags",
    "memory_kind",
    "confidence",
    "evidence_count",
    "consolidation_state",
    "sync_revision",
    "relationships",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _split(text: str) -> tuple[dict[str, Any], str]:
    normalized = text.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        return {}, normalized
    boundary = normalized.find("\n---\n", 4)
    if boundary < 0:
        return {}, normalized
    if yaml is None:
        raise RuntimeError("PyYAML is required; install the 'obsidian' extra")
    loaded = yaml.safe_load(normalized[4:boundary]) or {}
    if not isinstance(loaded, dict):
        loaded = {}
    return loaded, normalized[boundary + 5 :].lstrip("\n")


def _render(fields: dict[str, Any], body: str) -> str:
    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return json.dumps(
            value, ensure_ascii=False, separators=(", ", ": "), default=str
        )

    frontmatter = "\n".join(
        f"{key}: {scalar(value)}" for key, value in fields.items()
    )
    return f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n"


def _canonical_body(body: str) -> str:
    return re.sub(r"\s+", " ", body).strip().casefold()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:72] or "note"


def _link_filename(value: str) -> str:
    return (re.sub(r'[<>:"/\\|?*]', "-", value).strip(" .") or "Recovered")[:100]


def _active_durable(relative: Path, status: str) -> bool:
    parts = tuple(part.casefold() for part in relative.parts)
    if status != "active" or not parts:
        return False
    if parts[0] in {"maps", "inbox", "quarantine", "templates", "archive"}:
        return False
    if len(parts) >= 2 and parts[:2] in {
        ("system", "migration"),
        ("finance", "source"),
    }:
        return False
    return relative.as_posix().casefold() != "home.md"


def _primary_map(domain: str, relative: Path) -> str:
    value = f"{domain} {relative.as_posix()}".casefold()
    if "finance" in value:
        return "Finance"
    if any(token in value for token in ("project", "team", "area")):
        return "Projects"
    if any(token in value for token in ("people", "person", "contact")):
        return "People"
    if "decision" in value:
        return "Decisions"
    if any(token in value for token in ("research", "resource", "knowledge")):
        return "Research"
    return "Appliance and Systems"


def _brain_fields(
    fields: dict[str, Any], body: str, relative: Path, timestamp: str
) -> dict[str, Any]:
    """Normalize to the existing deterministic Hermes brain contract."""
    status = str(fields.get("status") or "active")
    if status not in {"active", "inactive", "completed", "archived", "quarantine"}:
        status = "quarantine" if status == "needs_review" else "active"
    sensitivity = str(fields.get("sensitivity") or "personal")
    if sensitivity not in {"public", "personal", "confidential", "financial"}:
        sensitivity = "personal"
    source_type = str(fields.get("source_type") or "legacy")
    if source_type not in {
        "conversation",
        "email",
        "email_fact",
        "manual",
        "legacy",
        "system",
        "redirect",
    }:
        source_type = "legacy"
    aliases = fields.get("aliases") if isinstance(fields.get("aliases"), list) else []
    tags = fields.get("tags") if isinstance(fields.get("tags"), list) else []
    def valid_timestamp(value: Any) -> str:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is not None:
                return parsed.astimezone(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                )
        except ValueError:
            pass
        return timestamp

    normalized = {
        **fields,
        "aliases": [str(value) for value in aliases if str(value).strip()],
        "status": status,
        "created": valid_timestamp(fields.get("created")),
        "updated": valid_timestamp(timestamp),
        "source_type": source_type,
        "source_ref": str(fields.get("source_ref") or f"migration:{relative.as_posix()}"),
        "sensitivity": sensitivity,
        "review_on": str(
            fields.get("review_on")
            or (datetime.now(timezone.utc).date() + timedelta(days=30)).isoformat()
        ),
        "tags": [str(value) for value in tags if str(value).strip()],
    }
    targets: list[tuple[str, str]] = []
    if _active_durable(relative, status):
        primary = _primary_map(str(normalized.get("domain", "")), relative)
        normalized["primary_map"] = primary
        targets.append((primary, "part-of"))
    else:
        normalized.pop("primary_map", None)
    for target in _LINK.findall(body):
        if target.casefold() not in {item[0].casefold() for item in targets}:
            targets.append((target, "related-to"))
    normalized["relationships"] = [
        f"{kind}|{target}|"
        + (
            "Primary domain map for this note."
            if kind == "part-of"
            else "Internal link retained during migration."
        )
        for target, kind in targets
    ]
    if sensitivity in {"confidential", "financial"}:
        normalized["processing"] = "local-only"
    return normalized


@dataclass(frozen=True)
class VaultAudit:
    notes: int
    invalid_frontmatter: int
    duplicate_ids: int
    exact_duplicate_groups: int
    unresolved_links: int
    orphans: int
    missing_primary_map: int
    missing_relationships: int
    valid: bool


@dataclass(frozen=True)
class MigrationReport:
    source: str
    staging: str
    notes_rewritten: int
    duplicates_archived: int
    link_stubs_created: int
    before: VaultAudit
    after: VaultAudit
    manifest: str


class VaultMigrator:
    """Repair a full vault in staging; cutover is an explicit separate action."""

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source).expanduser().resolve()
        if not self.source.is_dir():
            raise ValueError("vault source does not exist")

    @staticmethod
    def audit(root: str | Path) -> VaultAudit:
        root = Path(root).resolve()
        notes = sorted(root.rglob("*.md"))
        parsed: dict[Path, tuple[dict[str, Any], str]] = {}
        invalid = missing_map = missing_relationships = 0
        ids: list[str] = []
        targets: set[str] = set()
        inbound: Counter[str] = Counter()
        bodies: defaultdict[str, list[Path]] = defaultdict(list)
        for note in notes:
            targets.add(note.stem.casefold())
            targets.add(note.relative_to(root).with_suffix("").as_posix().casefold())
            try:
                fields, body = _split(note.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, ValueError, RuntimeError):
                fields, body = {}, ""
            parsed[note] = (fields, body)
            if not _REQUIRED.issubset(fields):
                invalid += 1
            if fields.get("id"):
                ids.append(str(fields["id"]))
            if _active_durable(
                note.relative_to(root), str(fields.get("status") or "active")
            ) and not fields.get("primary_map"):
                missing_map += 1
            if "relationships" not in fields:
                missing_relationships += 1
            bodies[_canonical_body(body)].append(note)
        unresolved = 0
        for _, body in parsed.values():
            for target in _LINK.findall(body):
                key = target.strip().replace("\\", "/").casefold()
                if key not in targets and Path(key).name not in targets:
                    unresolved += 1
                else:
                    inbound[Path(key).name] += 1
        duplicate_ids = sum(count - 1 for count in Counter(ids).values() if count > 1)
        duplicate_groups = sum(
            1 for body, paths in bodies.items() if body and len(paths) > 1
        )
        orphans = sum(
            1
            for note in notes
            if inbound[note.stem.casefold()] == 0
            and note.as_posix().casefold().find("/maps/") < 0
        )
        valid = (
            invalid == 0
            and duplicate_ids == 0
            and duplicate_groups == 0
            and unresolved == 0
            and missing_map == 0
            and missing_relationships == 0
        )
        return VaultAudit(
            len(notes),
            invalid,
            duplicate_ids,
            duplicate_groups,
            unresolved,
            orphans,
            missing_map,
            missing_relationships,
            valid,
        )

    def stage(self, destination: str | Path) -> MigrationReport:
        destination = Path(destination).expanduser().resolve()
        if destination.exists():
            raise ValueError("staging destination must not already exist")
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = self.audit(self.source)
        shutil.copytree(
            self.source,
            destination,
            ignore=shutil.ignore_patterns(".trash", ".git", "*.tmp"),
        )
        notes = sorted(destination.rglob("*.md"))
        seen_ids: set[str] = set()
        seen_bodies: dict[str, tuple[str, Path]] = {}
        rewritten = archived = 0
        manifest_entries: list[dict[str, Any]] = []
        timestamp = _now()
        for note in notes:
            relative = note.relative_to(destination)
            original = note.read_text(encoding="utf-8")
            fields, body = _split(original)
            stable_id = str(uuid.uuid5(_NAMESPACE, relative.as_posix().casefold()))
            note_id = str(fields.get("id") or stable_id)
            if note_id in seen_ids:
                fields["previous_id"] = note_id
                note_id = stable_id
            seen_ids.add(note_id)
            stat = note.stat()
            created = datetime.fromtimestamp(
                stat.st_ctime, timezone.utc
            ).isoformat().replace("+00:00", "Z")
            domain = relative.parts[0] if len(relative.parts) > 1 else "General"
            fields = {
                **fields,
                "id": note_id,
                "title": str(fields.get("title") or note.stem),
                "aliases": fields.get("aliases") or [],
                "type": str(fields.get("type") or "note"),
                "domain": str(fields.get("domain") or domain),
                "status": str(fields.get("status") or "active"),
                "created": fields.get("created") or created,
                "updated": timestamp,
                "source_type": str(fields.get("source_type") or "imported"),
                "source_ref": str(
                    fields.get("source_ref") or f"obsidian:{relative.as_posix()}"
                ),
                "sensitivity": str(fields.get("sensitivity") or "internal"),
                "review_on": fields.get("review_on"),
                "tags": fields.get("tags") or [],
                "memory_kind": str(fields.get("memory_kind") or "fact"),
                "confidence": float(fields.get("confidence") or 0.5),
                "evidence_count": int(fields.get("evidence_count") or 1),
                "consolidation_state": str(
                    fields.get("consolidation_state") or "imported"
                ),
                "sync_revision": int(fields.get("sync_revision") or 0),
                "primary_map": fields.get("primary_map") or "[[Maps/Home]]",
                "relationships": fields.get("relationships") or [],
            }
            canonical = _canonical_body(body)
            destination_note = note
            if canonical and canonical in seen_bodies:
                original_id, original_note = seen_bodies[canonical]
                fields["status"] = "archived"
                fields["duplicate_of"] = original_id
                fields["archive_reason"] = "exact_duplicate"
                destination_note = (
                    destination
                    / "Archive"
                    / "Duplicates"
                    / f"{_slug(note.stem)}-dup-{stable_id[:8]}.md"
                )
                destination_note.parent.mkdir(parents=True, exist_ok=True)
                archived += 1
                body += (
                    f"\n\n> Archived duplicate of "
                    f"[[{original_note.relative_to(destination).with_suffix('').as_posix()}]]."
                    f"\n> Archive record: `{stable_id}`."
                )
            elif canonical:
                seen_bodies[canonical] = (note_id, note)
            fields = _brain_fields(
                fields,
                body,
                destination_note.relative_to(destination),
                timestamp,
            )
            rendered = _render(fields, body)
            destination_note.write_text(rendered, encoding="utf-8", newline="\n")
            if destination_note != note:
                note.unlink()
            rewritten += 1
            manifest_entries.append(
                {
                    "source": relative.as_posix(),
                    "destination": destination_note.relative_to(
                        destination
                    ).as_posix(),
                    "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
                    "after_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "memory_id": note_id,
                }
            )

        self._rename_generic_readmes(destination, timestamp)
        self._ensure_canonical_maps(destination, timestamp)
        stubs = self._repair_links(destination, timestamp)
        after = self.audit(destination)
        manifest_path = destination / ".hippocampal-memory" / "migration.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        report = MigrationReport(
            str(self.source),
            str(destination),
            rewritten,
            archived,
            stubs,
            before,
            after,
            str(manifest_path),
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "created_at": timestamp,
                    "report": {
                        **asdict(report),
                        "before": asdict(before),
                        "after": asdict(after),
                    },
                    "mutations": manifest_entries,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _rename_generic_readmes(destination: Path, timestamp: str) -> int:
        renamed = 0
        for note in sorted(destination.rglob("*.md")):
            if note.name.casefold() != "readme.md":
                continue
            relative = note.relative_to(destination)
            fields, body = _split(note.read_text(encoding="utf-8"))
            title = str(fields.get("title") or note.parent.name or "Overview")
            filename = _link_filename(title)
            if filename.casefold() == "readme":
                filename = _link_filename(note.parent.name + " Overview")
            target = note.with_name(filename + ".md")
            if target.exists() and target != note:
                target = note.with_name(
                    f"{filename}-{str(fields.get('id') or uuid.uuid4())[:8]}.md"
                )
            aliases = fields.get("aliases")
            if not isinstance(aliases, list):
                aliases = []
            old_target = relative.with_suffix("").as_posix()
            for alias in ("README", old_target):
                if alias.casefold() not in {
                    str(value).casefold() for value in aliases
                }:
                    aliases.append(alias)
            fields["aliases"] = aliases
            new_relative = target.relative_to(destination)
            fields = _brain_fields(fields, body, new_relative, timestamp)
            target.write_text(_render(fields, body), encoding="utf-8", newline="\n")
            note.unlink()
            renamed += 1
        return renamed

    @staticmethod
    def _ensure_canonical_maps(destination: Path, timestamp: str) -> None:
        maps = destination / "Maps"
        maps.mkdir(parents=True, exist_ok=True)
        map_names = (
            "Finance",
            "Projects",
            "Research",
            "People",
            "Decisions",
            "Appliance and Systems",
        )

        def write_system_note(relative: Path, title: str, body: str) -> None:
            path = destination / relative
            if path.exists():
                return
            fields = _brain_fields(
                {
                    "id": str(uuid.uuid5(_NAMESPACE, relative.as_posix().casefold())),
                    "title": title,
                    "aliases": [],
                    "type": "map",
                    "domain": "System",
                    "status": "active",
                    "created": timestamp,
                    "updated": timestamp,
                    "source_type": "system",
                    "source_ref": "vault-migration:canonical-map",
                    "sensitivity": "personal",
                    "review_on": None,
                    "tags": ["map"],
                    "memory_kind": "fact",
                    "confidence": 1.0,
                    "evidence_count": 1,
                    "consolidation_state": "system",
                    "sync_revision": 0,
                    "relationships": [],
                },
                body,
                relative,
                timestamp,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(fields, body), encoding="utf-8", newline="\n")

        for name in map_names:
            write_system_note(
                Path("Maps") / f"{name}.md",
                name,
                f"# {name}\n\nDomain map maintained by Hermes.\n\n- [[Home]]",
            )
        write_system_note(
            Path("Home.md"),
            "Home",
            "# Home\n\n"
            + "\n".join(f"- [[Maps/{name}]]" for name in map_names),
        )

    @staticmethod
    def _repair_links(destination: Path, timestamp: str) -> int:
        notes = sorted(destination.rglob("*.md"))
        documents: dict[str, tuple[Path, dict[str, Any], str]] = {}
        aliases: defaultdict[str, list[str]] = defaultdict(list)
        for note in notes:
            relative = note.relative_to(destination).as_posix()
            fields, body = _split(note.read_text(encoding="utf-8"))
            documents[relative] = (note, fields, body)
            path = Path(relative)
            keys = {
                path.stem.casefold(),
                path.with_suffix("").as_posix().casefold(),
                str(fields.get("title") or "").strip().casefold(),
            }
            keys.update(
                str(value).strip().casefold()
                for value in fields.get("aliases", [])
                if str(value).strip()
            )
            for key in keys - {""}:
                aliases[key].append(relative)

        missing: dict[str, str] = {}
        for relative, (note, fields, body) in documents.items():
            source_path = Path(relative)

            def replace(match: re.Match[str]) -> str:
                target = match.group(1).strip().replace("\\", "/")
                heading = match.group(2) or ""
                display = match.group(3)
                key = target.removesuffix(".md").casefold()
                if (
                    re.search(r'[<>:"|?*\uf000-\uf0ff]', target)
                    or " / " in target
                    or key.startswith("finance/source/")
                ):
                    return display or target
                matches = aliases.get(key, [])
                chosen: str | None = matches[0] if len(matches) == 1 else None
                local = (
                    source_path.parent / (target.removesuffix(".md") + ".md")
                ).as_posix()
                local = Path(local).as_posix()
                if local in documents:
                    chosen = local
                if chosen is None and target.startswith("../"):
                    candidate = (
                        note.parent / (target.removesuffix(".md") + ".md")
                    ).resolve()
                    try:
                        chosen_relative = candidate.relative_to(destination).as_posix()
                    except ValueError:
                        chosen_relative = ""
                    if chosen_relative in documents:
                        chosen = chosen_relative
                if chosen is None and source_path.parts[0].casefold() == "team":
                    prefixed = (
                        Path("Team") / (target.removesuffix(".md") + ".md")
                    ).as_posix()
                    if prefixed in documents or target.split("/", 1)[0] in {
                        "Audits",
                        "System",
                        "Projects",
                        "Context",
                    }:
                        chosen = prefixed
                        if prefixed not in documents:
                            missing[prefixed] = target
                if chosen is None and len(matches) > 1:
                    # Ambiguous links must become explicit paths. Prefer a
                    # candidate sharing the longest directory prefix.
                    source_parts = source_path.parent.parts
                    chosen = max(
                        matches,
                        key=lambda value: sum(
                            left.casefold() == right.casefold()
                            for left, right in zip(
                                source_parts, Path(value).parent.parts
                            )
                        ),
                    )
                if chosen is None:
                    if "/" in target:
                        chosen = target.removesuffix(".md") + ".md"
                    else:
                        chosen = f"Recovered Links/{_link_filename(target)}.md"
                    missing[chosen] = target
                link_target = Path(chosen).with_suffix("").as_posix()
                alias = f"|{display}" if display else ""
                return f"[[{link_target}{heading}{alias}]]"

            rewritten = _FULL_LINK.sub(replace, body)
            if rewritten != body:
                fields = _brain_fields(
                    fields, rewritten, Path(relative), timestamp
                )
                note.write_text(
                    _render(fields, rewritten), encoding="utf-8", newline="\n"
                )

        created = 0
        for relative_string, original_target in sorted(missing.items()):
            relative = Path(relative_string)
            path = destination / relative
            if path.exists():
                continue
            body = (
                f"# {original_target}\n\nRecovered placeholder for a previously "
                "unresolved Obsidian link. Replace it with sourced information "
                "or archive it."
            )
            fields = _brain_fields(
                {
                    "id": str(
                        uuid.uuid5(
                            _NAMESPACE, "recovered:" + relative.as_posix().casefold()
                        )
                    ),
                    "title": original_target,
                    "aliases": [],
                    "type": "reference_stub",
                    "domain": "Recovered Links",
                    "status": "quarantine",
                    "created": timestamp,
                    "updated": timestamp,
                    "source_type": "system",
                    "source_ref": "vault-migration:unresolved-link",
                    "sensitivity": "personal",
                    "review_on": None,
                    "tags": ["recovered-link"],
                    "memory_kind": "fact",
                    "confidence": 0.1,
                    "evidence_count": 0,
                    "consolidation_state": "unverified",
                    "sync_revision": 0,
                    "relationships": [],
                },
                body,
                relative,
                timestamp,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(_render(fields, body), encoding="utf-8", newline="\n")
            created += 1
        return created

    def cutover(self, staging: str | Path, archive: str | Path) -> dict[str, str]:
        """Atomically exchange the staged vault only after a valid audit."""
        staging = Path(staging).resolve()
        archive = Path(archive).resolve()
        if not staging.is_dir() or archive.exists():
            raise ValueError("staging must exist and archive must not exist")
        if not self.audit(staging).valid:
            raise RuntimeError("staged vault did not pass acceptance audit")
        if staging.parent != self.source.parent:
            raise ValueError("staging and live vault must share a volume and parent")
        self.source.rename(archive)
        try:
            staging.rename(self.source)
        except Exception:
            archive.rename(self.source)
            raise
        return {
            "status": "committed",
            "live": str(self.source),
            "archive": str(archive),
        }
