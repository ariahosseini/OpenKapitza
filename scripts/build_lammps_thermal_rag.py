"""Assemble a RAG corpus for LAMMPS thermal analysis from the official sources.

Scope, deliberately narrow: the thermal-transport path only. The five official
`examples/KAPPA` input scripts, the Howto page that frames them, the kappa
extraction recipes from the KAPPA README, and the reference pages for exactly
the commands those scripts use. Nothing else from the manual.

WHY A GENERATOR RATHER THAN A HAND-WRITTEN FILE

docs.lammps.org is built from the `.rst` files in the LAMMPS source tree, so a
checkout is the same content as the website plus two advantages: it is pinned
to the version of the binary actually being run, and it needs no scraping.
Copying 450 lines of input script by hand is exactly the transcription risk
this project keeps getting burned by -- a fabricated `SiGe.sw`, an element
label guessed as `Si` where the file says `Si(D)`, a correct formula nearly
"fixed" from a bad PDF read. So nothing here is retyped: every script is
emitted verbatim from the source tree, and every doc extract is quoted from
its `.rst` with the file and version recorded.

Regenerate after a LAMMPS upgrade rather than editing the output.

Usage:
    python3 scripts/build_lammps_thermal_rag.py --lammps-src <path to clone>
                                                [--out docs/lammps-thermal-rag.md]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Only commands that appear in the KAPPA scripts, plus the two the interface
# workflow adds (chunk binning is shared; temp/region is how the manual
# thermostats a slab). Keeping this list short is the point.
COMMAND_PAGES = [
    ("fix_langevin", "fix langevin", "Langevin reservoirs (method 1)"),
    ("fix_heat", "fix heat", "add/subtract energy (method 2)"),
    ("fix_ehex", "fix ehex", "enhanced heat exchange (method 3)"),
    ("fix_thermal_conductivity", "fix thermal/conductivity",
     "Muller-Plathe reverse NEMD (method 4)"),
    ("compute_heat_flux", "compute heat/flux", "Green-Kubo (method 5)"),
    ("compute_temp_region", "compute temp/region",
     "temperature of a spatial slab"),
    ("compute_ke_atom", "compute ke/atom", "per-atom KE for profiles"),
    ("compute_chunk_atom", "compute chunk/atom", "binning along an axis"),
    ("fix_ave_chunk", "fix ave/chunk", "temperature profile"),
    ("fix_ave_time", "fix ave/time", "running averages"),
    ("fix_ave_correlate", "fix ave/correlate", "autocorrelation for Green-Kubo"),
]

SCRIPTS = [
    ("in.langevin", "Method 1 - thermostat two slabs with fix langevin"),
    ("in.heat", "Method 2 - add/subtract energy with fix heat"),
    ("in.ehex", "Method 3 - enhanced heat exchange"),
    ("in.mp", "Method 4 - Muller-Plathe reverse NEMD"),
    ("in.heatflux", "Method 5 - Green-Kubo equilibrium"),
]


def lammps_version(src: Path) -> str:
    header = src / "src" / "version.h"
    if header.is_file():
        m = re.search(r'"([^"]+)"', header.read_text())
        if m:
            return m.group(1)
    return "unknown"


def derst(text: str) -> str:
    """Minimal reStructuredText -> Markdown, faithful rather than complete.

    Only the constructs that appear in these pages are handled. Anything not
    recognised is left as-is: a stray directive in the corpus is harmless,
    whereas silently dropping text would make the quote inaccurate.
    """
    text = re.sub(r":doc:`([^`<]+?)\s*<[^>]+>`", r"`\1`", text)
    text = re.sub(r":doc:`([^`]+?)`", r"`\1`", text)
    text = re.sub(r":ref:`([^`<]+?)\s*<[^>]+>`", r"\1", text)
    text = re.sub(r":math:`([^`]+?)`", r"$\1$", text)
    text = re.sub(r":doc:|:ref:", "", text)
    text = text.replace("\\ ", "")
    text = re.sub(r"\*([A-Za-z_/][A-Za-z0-9_/\- ]*)\*", r"*\1*", text)
    return text


def extract_sections(rst: str, wanted: tuple[str, ...]) -> dict[str, str]:
    """Pull named sections, recognised by an underline of punctuation."""
    lines = rst.splitlines()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for i, line in enumerate(lines):
        if (i + 1 < len(lines) and line.strip()
                and set(lines[i + 1].strip()) <= set('"\'=-~^#*+`')
                and len(lines[i + 1].strip()) >= max(3, len(line.strip()) - 2)):
            title = line.strip()
            current = title if title in wanted else None
            if current:
                sections[current] = []
            continue
        if current and sections.get(current) is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def code_blocks(section: str) -> list[str]:
    """The `.. code-block:: LAMMPS` bodies inside a section."""
    blocks: list[str] = []
    lines = section.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"\s*\.\.\s+code-block::", lines[i]):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            body: list[str] = []
            indent = None
            while i < len(lines):
                line = lines[i]
                if not line.strip():
                    body.append("")
                    i += 1
                    continue
                current_indent = len(line) - len(line.lstrip())
                if indent is None:
                    indent = current_indent
                if current_indent < indent:
                    break
                body.append(line[indent:])
                i += 1
            text = "\n".join(body).strip("\n")
            if text:
                blocks.append(text)
            continue
        i += 1
    return blocks


def admonitions(rst: str) -> list[tuple[str, str]]:
    """`.. note::` and `.. warning::` bodies -- where the gotchas live."""
    found: list[tuple[str, str]] = []
    lines = rst.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"\s*\.\.\s+(note|warning)::\s*$", lines[i])
        if not m:
            i += 1
            continue
        kind = m.group(1)
        base = len(lines[i]) - len(lines[i].lstrip())
        i += 1
        body: list[str] = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= base:
                break
            body.append(line.strip())
            i += 1
        text = " ".join(x for x in body if x).strip()
        if text:
            found.append((kind, text))
    return found


def build(src: Path, out: Path) -> int:
    kappa = src / "examples" / "KAPPA"
    docs = src / "doc" / "src"
    if not kappa.is_dir() or not docs.is_dir():
        print(f"not a LAMMPS source tree: {src}")
        return 2

    version = lammps_version(src)
    parts: list[str] = []
    add = parts.append

    add("# LAMMPS thermal-analysis reference (RAG corpus)\n")
    add("> **Generated — do not edit.** Rebuild with\n"
        "> `python3 scripts/build_lammps_thermal_rag.py --lammps-src <clone>`.\n>\n"
        f"> Source: official LAMMPS source tree, version **{version}**.\n"
        "> `docs.lammps.org` is built from these same `.rst` files, so this is\n"
        "> the manual's content pinned to the engine actually in use. Scripts\n"
        "> are emitted verbatim; prose is quoted with its source file named.\n")
    add("**Scope:** the thermal-transport path only — the five official "
        "`examples/KAPPA` scripts, the Howto that frames them, the kappa "
        "extraction recipes, and reference pages for exactly the commands "
        "those scripts use. Nothing else from the manual.\n")
    add("**Retrieval note.** Each command below carries its Syntax block, its "
        "official examples, and every `note`/`warning` admonition from its "
        "page. The admonitions are the highest-value part: they are where the "
        "manual records the constraints that otherwise surface as a failed "
        "run.\n")
    add("---\n")

    # 1. The Howto that frames the methods.
    howto = docs / "Howto_kappa.rst"
    if howto.is_file():
        add("## Overview: calculating thermal conductivity\n")
        add(f"*Source: `doc/src/Howto_kappa.rst`*\n")
        body = derst(howto.read_text())
        body = re.sub(r"^Calculate thermal conductivity\s*\n=+\s*\n", "", body)
        body = re.sub(r"\n-{4,}\n.*$", "", body, flags=re.S)
        add(body.strip() + "\n")
        add("---\n")

    # 2. How to get kappa out, straight from the README.
    readme = kappa / "README"
    if readme.is_file():
        add("## Extracting kappa from each method (official recipes)\n")
        add("*Source: `examples/KAPPA/README`, verbatim*\n")
        add("```text\n" + readme.read_text().strip() + "\n```\n")
        add("---\n")

    # 3. The scripts themselves, verbatim.
    add("## The five official scripts\n")
    add("*Source: `examples/KAPPA/`, verbatim. The README notes these are "
        "illustrative: \"No guarantee is made that the systems are fully "
        "equilibrated or that the runs are long enough to generate good "
        "statistics.\"*\n")
    for name, caption in SCRIPTS:
        path = kappa / name
        if not path.is_file():
            continue
        add(f"### `{name}` — {caption}\n")
        add("```lammps\n" + path.read_text().rstrip() + "\n```\n")
    add("---\n")

    # 4. Reference pages for the commands those scripts use.
    add("## Command reference\n")
    wanted = ("Syntax", "Examples", "Description")
    for stem, command, role in COMMAND_PAGES:
        path = docs / f"{stem}.rst"
        if not path.is_file():
            continue
        rst = path.read_text()
        sections = extract_sections(rst, wanted)
        add(f"### `{command}` — {role}\n")
        add(f"*Source: `doc/src/{stem}.rst`*\n")

        syntax = sections.get("Syntax", "")
        for block in code_blocks(syntax)[:1]:
            add("**Syntax**\n\n```lammps\n" + block + "\n```\n")
        # Keep the bullet list that defines each argument.
        bullets = [l for l in syntax.splitlines() if l.strip().startswith("*")]
        if bullets:
            add("\n".join(derst(l).rstrip() for l in bullets[:14]) + "\n")

        for block in code_blocks(sections.get("Examples", ""))[:1]:
            add("**Examples**\n\n```lammps\n" + block + "\n```\n")

        description = sections.get("Description", "")
        if description:
            # Drop subsection underlines ("""""" etc.) that belong to nested
            # headings, and directive lines, so the quoted prose reads clean.
            cleaned = "\n".join(
                line for line in derst(description).splitlines()
                if not (line.strip() and set(line.strip()) <= set('"\'=-~^#*+`'))
            )
            prose = [p.strip() for p in cleaned.split("\n\n")
                     if p.strip() and not p.strip().startswith("..")]
            if prose:
                add("**Description** (first paragraphs)\n")
                add("\n\n".join(prose[:3]) + "\n")

        notes = admonitions(rst)
        if notes:
            add("**Constraints and cautions from this page**\n")
            for kind, text in notes:
                add(f"- **{kind.upper()}:** {derst(text)}")
            add("")
    add("---\n")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts))
    size_kb = out.stat().st_size / 1024
    print(f"wrote {out} ({size_kb:.0f} KB, {len(out.read_text().splitlines())} lines)")
    print(f"LAMMPS version: {version}")
    print(f"scripts: {sum(1 for n, _ in SCRIPTS if (kappa / n).is_file())}/"
          f"{len(SCRIPTS)}   command pages: "
          f"{sum(1 for s, _, _ in COMMAND_PAGES if (docs / f'{s}.rst').is_file())}"
          f"/{len(COMMAND_PAGES)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-src", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "docs"
        / "lammps-thermal-rag.md")
    args = parser.parse_args(argv)
    return build(args.lammps_src, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
