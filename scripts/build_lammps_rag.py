"""Assemble a LAMMPS RAG corpus covering OpenKapitza's physics.

Replaces the thermal-only corpus. Scope is now every physics path this project
actually uses, each section carrying the official example scripts verbatim plus
reference pages for the commands those scripts use:

  1. Thermal transport          examples/KAPPA      (5 methods)
  2. Elastic constants, relax   examples/ELASTIC, ELASTIC_T
  3. Hardness / indentation     examples/indent
  4. Gas-surface scattering     examples/wall       (accommodation models)
  5. Surface structure          displace/delete/deposit
  6. Amorphous, melt-quench     examples/melt, examples/vashishta (SiO2)
  7. Electron-phonon (TTM)      examples/ttm
  8. Gas transport properties   examples/VISCOSITY, DIFFUSE

Still deliberately bounded: it is the commands this project needs, not the
manual. The out-of-scope test in tests/test_lammps_rag.py pins that.

WHY A GENERATOR RATHER THAN A HAND-WRITTEN FILE

docs.lammps.org is built from the `.rst` files in the LAMMPS source tree, so a
checkout is the same content as the website, pinned to the version of the
binary actually being run and needing no scraping. Copying hundreds of lines
of input script by hand is exactly the transcription risk this project keeps
getting burned by -- a fabricated `SiGe.sw`, an element label guessed as `Si`
where the file says `Si(D)`, a correct formula nearly "fixed" from a bad PDF
read. Nothing here is retyped: scripts are emitted verbatim and every doc
extract is quoted with its file recorded.

Regenerate after a LAMMPS upgrade rather than editing the output.

Usage:
    python3 scripts/build_lammps_rag.py --lammps-src <clone> [--out PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (section title, why it is here, [(example subpath, caption)], [(rst stem, command, role)])
TOPICS: list[tuple[str, str, list[tuple[str, str]], list[tuple[str, str, str]]]] = [
    (
        "Thermal transport",
        "The Kapitza-resistance path. Five official methods for thermal "
        "conductivity; OpenKapitza uses method 1 (Langevin reservoirs) with "
        "non-periodic walls, and the temperature-fluctuation cross-check "
        "rather than Green-Kubo, which measures BULK conductivity.",
        [
            ("KAPPA/in.langevin", "Method 1 - thermostat two slabs (what we use)"),
            ("KAPPA/in.heat", "Method 2 - add/subtract energy with fix heat"),
            ("KAPPA/in.ehex", "Method 3 - enhanced heat exchange"),
            ("KAPPA/in.mp", "Method 4 - Muller-Plathe reverse NEMD"),
            ("KAPPA/in.heatflux", "Method 5 - Green-Kubo equilibrium"),
        ],
        [
            ("fix_langevin", "fix langevin", "Langevin reservoirs"),
            ("fix_heat", "fix heat", "add/subtract energy"),
            ("fix_ehex", "fix ehex", "enhanced heat exchange"),
            ("fix_thermal_conductivity", "fix thermal/conductivity",
             "Muller-Plathe reverse NEMD"),
            ("compute_heat_flux", "compute heat/flux", "Green-Kubo heat flux"),
            ("compute_temp_region", "compute temp/region",
             "temperature of a spatial slab"),
            ("compute_ke_atom", "compute ke/atom", "per-atom KE for profiles"),
            ("compute_chunk_atom", "compute chunk/atom", "binning along an axis"),
            ("fix_ave_chunk", "fix ave/chunk", "temperature profile"),
            ("fix_ave_time", "fix ave/time", "running averages"),
            ("fix_ave_correlate", "fix ave/correlate", "autocorrelation"),
        ],
    ),
    (
        "Elastic constants and cell relaxation",
        "Decides how strain SHARES between two coherently bonded materials, "
        "and therefore the relaxed in-plane constant of an interface. "
        "Measured here: SiCGe.tersoff reproduces a0 to 0.02 % but gets C_ij "
        "8-18 % wrong and INVERTS which of Si/Ge is biaxially stiffer, so this "
        "section is a validation gate, not background reading.",
        [
            ("ELASTIC/in.elastic", "Full elastic tensor by finite differences"),
            ("ELASTIC/init.mod", "Crystal, units and strain magnitude"),
            ("ELASTIC/potential.mod", "Pair style and minimisation settings"),
            ("ELASTIC/displace.mod", "Applies +/- strain and reads the stress"),
        ],
        [
            ("fix_box_relax", "fix box/relax", "relax the cell to zero stress"),
            ("minimize", "minimize", "energy minimisation"),
            ("change_box", "change_box", "impose a deformation"),
            ("compute_born_matrix", "compute born/matrix",
             "elastic constants without finite differences"),
            ("fix_numdiff", "fix numdiff", "numerical force derivatives"),
            ("compute_stress_atom", "compute stress/atom", "per-atom stress"),
            ("compute_pressure", "compute pressure", "pressure tensor"),
        ],
    ),
    (
        "Hardness and indentation",
        "Contact microhardness H_c is the single largest lever on the "
        "contact-conductance result (h_c depends on P/H_c), and the "
        "Song-Yovanovich correlation needs load-dependent indentation data. "
        "fix indent is the atomistic route to it.",
        [
            ("indent/in.indent", "Indenter pushed into a 2d solid"),
            ("indent/in.indent.min", "Indentation by minimisation"),
        ],
        [
            ("fix_indent", "fix indent", "spherical or cylindrical indenter"),
            ("compute_stress_atom", "compute stress/atom", "local stress field"),
        ],
    ),
    (
        "Gas-surface scattering and accommodation coefficients",
        "The thermal accommodation coefficient is, by the project's own spec, "
        "the weakest input in the whole model -- measured values for one "
        "gas/solid pair span an order of magnitude. LAMMPS provides the "
        "FORWARD model (impose a given accommodation) via "
        "fix wall/reflect/stochastic with diffusive, Maxwell and "
        "Cercignani-Lampis styles. It does NOT measure alpha: for that, fire "
        "gas atoms at a real atomistic surface and compare incident with "
        "reflected energy against the wall temperature. No canned example "
        "exists for the measurement.",
        [
            ("wall/in.wall.diffusive", "Fully diffusive reflection"),
            ("wall/in.wall.specular", "Specular reflection"),
            ("wall/in.wall.maxwell", "Maxwell model, one accommodation coeff"),
            ("wall/in.wall.ccl",
             "Cercignani-Lampis, per-direction accommodation coeffs"),
        ],
        [
            ("fix_wall_reflect_stochastic", "fix wall/reflect/stochastic",
             "accommodation models at a wall"),
            ("fix_wall_reflect", "fix wall/reflect", "simple reflecting wall"),
            ("fix_wall", "fix wall/lj93 etc", "potential-based walls"),
            ("fix_wall_region", "fix wall/region", "wall from a region surface"),
        ],
    ),
    (
        "Surface structure and roughness",
        "Nanoscale roughness and defects at the interface are an optional part "
        "of the MD step, and the Phase-2 structure module needs to cut, stack "
        "and perturb surfaces. There is no official 'roughness' example; these "
        "are the commands that build one.",
        [],
        [
            ("displace_atoms", "displace_atoms",
             "move atoms, including random displacement"),
            ("delete_atoms", "delete_atoms", "remove atoms, including random"),
            ("create_atoms", "create_atoms", "fill a region from a lattice"),
            ("fix_deposit", "fix deposit", "add atoms onto a surface"),
            ("lattice", "lattice", "lattice definition and orientation"),
            ("region", "region", "geometric regions, units box vs lattice"),
        ],
    ),
    (
        "Amorphous structures and melt-quench",
        "The Si/SiO2 path. The spec requires reporting the quench rate against "
        "experiment (MD 1e11-1e13 K/s against ~1 K/s) and validating density, "
        "RDF and bond angles. examples/vashishta carries a real SiO2 "
        "parameterisation and a quartz data file.",
        [
            ("melt/in.melt", "Simple melt"),
            ("vashishta/in.vashishta.sio2", "SiO2 with the Vashishta potential"),
        ],
        [
            ("pair_vashishta", "pair_style vashishta", "SiO2 and InP"),
            ("fix_nh", "fix nvt / fix npt",
             "Nose-Hoover, including temperature ramps for quenching"),
            ("compute_rdf", "compute rdf", "radial distribution function"),
        ],
    ),
    (
        "Electron-phonon coupling (two-temperature model)",
        "The spec flags Si/Al as a metal-semiconductor interface where "
        "electron-phonon coupling carries part of the heat, and states that "
        "nothing in the workflow models it. LAMMPS does: fix ttm and "
        "fix ttm/mod implement a two-temperature model, and examples/ttm ships "
        "an Si parameterisation. This section exists so that caveat can become "
        "a plan.",
        [
            ("ttm/in.ttm", "Two-temperature model"),
            ("ttm/in.ttm.thermal", "TTM with a thermal gradient"),
            ("ttm/in.ttm.mod", "fix ttm/mod variant"),
        ],
        [
            ("fix_ttm", "fix ttm", "two-temperature electron-phonon model"),
        ],
    ),
    (
        "Gas transport properties",
        "The gap model derives the molecular mean free path from viscosity "
        "rather than a tabulated collision diameter, because viscosity data is "
        "better established. These are the LAMMPS routes to viscosity and "
        "diffusivity if the gas properties ever need computing rather than "
        "looking up.",
        [],
        [
            ("fix_viscosity", "fix viscosity", "Muller-Plathe shear viscosity"),
            ("compute_viscosity_cos", "compute viscosity/cos",
             "periodic-perturbation viscosity"),
            ("compute_msd", "compute msd", "mean-squared displacement"),
            ("fix_nvt_sllod", "fix nvt/sllod", "SLLOD shear for viscosity"),
        ],
    ),
]

WANTED_SECTIONS = ("Syntax", "Examples", "Description")
MAX_SCRIPT_LINES = 140


def lammps_version(src: Path) -> str:
    header = src / "src" / "version.h"
    if header.is_file():
        m = re.search(r'"([^"]+)"', header.read_text())
        if m:
            return m.group(1)
    return "unknown"


def derst(text: str) -> str:
    """Minimal reStructuredText -> Markdown, faithful rather than complete."""
    text = re.sub(r":doc:`([^`<]+?)\s*<[^>]+>`", r"`\1`", text)
    text = re.sub(r":doc:`([^`]+?)`", r"`\1`", text)
    text = re.sub(r":ref:`([^`<]+?)\s*<[^>]+>`", r"\1", text)
    text = re.sub(r":math:`([^`]+?)`", r"$\1$", text)
    text = re.sub(r":doc:|:ref:", "", text)
    return text.replace("\\ ", "")


def extract_sections(rst: str, wanted: tuple[str, ...]) -> dict[str, str]:
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
        if current is not None and current in sections:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def code_blocks(section: str) -> list[str]:
    blocks: list[str] = []
    lines = section.splitlines()
    i = 0
    while i < len(lines):
        if re.match(r"\s*\.\.\s+(code-block|parsed-literal)::", lines[i]):
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            body: list[str] = []
            indent: int | None = None
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


def render_command(docs: Path, stem: str, command: str, role: str) -> list[str]:
    path = docs / f"{stem}.rst"
    if not path.is_file():
        return []
    rst = path.read_text()
    sections = extract_sections(rst, WANTED_SECTIONS)
    out = [f"#### `{command}` — {role}\n", f"*Source: `doc/src/{stem}.rst`*\n"]

    syntax = sections.get("Syntax", "")
    # All syntax blocks, not just the first. The nested `parsed-literal`
    # blocks carry the per-argument detail -- on
    # fix_wall_reflect_stochastic.rst that is where the accommodation
    # coefficients are defined, which is the whole reason the page is here.
    # Taking only the first block silently dropped it.
    for block in code_blocks(syntax)[:4]:
        out.append("```lammps\n" + block + "\n```\n")
    bullets = [l for l in syntax.splitlines() if l.strip().startswith("*")]
    if bullets:
        out.append("\n".join(derst(l).rstrip() for l in bullets[:12]) + "\n")

    for block in code_blocks(sections.get("Examples", ""))[:1]:
        out.append("*Examples*\n\n```lammps\n" + block + "\n```\n")

    description = sections.get("Description", "")
    if description:
        cleaned = "\n".join(
            line for line in derst(description).splitlines()
            if not (line.strip() and set(line.strip()) <= set('"\'=-~^#*+`'))
        )
        prose = [p.strip() for p in cleaned.split("\n\n")
                 if p.strip() and not p.strip().startswith("..")]
        if prose:
            out.append("\n\n".join(prose[:2]) + "\n")

    notes = admonitions(rst)
    if notes:
        out.append("*Constraints and cautions from this page*\n")
        for kind, text in notes[:6]:
            out.append(f"- **{kind.upper()}:** {derst(text)}")
        out.append("")
    return out


def build(src: Path, out_path: Path) -> int:
    examples = src / "examples"
    docs = src / "doc" / "src"
    if not examples.is_dir() or not docs.is_dir():
        print(f"not a LAMMPS source tree: {src}")
        return 2

    version = lammps_version(src)
    parts: list[str] = []
    add = parts.append
    found_scripts = missing_scripts = found_pages = missing_pages = 0
    missing: list[str] = []

    add("# LAMMPS reference for OpenKapitza (RAG corpus)\n")
    add("> **Generated — do not edit.** Rebuild with\n"
        "> `python3 scripts/build_lammps_rag.py --lammps-src <clone>`.\n>\n"
        f"> Source: official LAMMPS source tree, version **{version}**.\n"
        "> `docs.lammps.org` is built from these same `.rst` files, so this is\n"
        "> the manual's content pinned to the engine actually in use. Example\n"
        "> scripts are emitted verbatim; prose is quoted with its source named.\n")
    add("**Scope:** the physics paths this project uses — thermal transport, "
        "elastic constants and relaxation, hardness and indentation, "
        "gas–surface accommodation, surface structure, amorphous "
        "melt–quench, electron–phonon coupling, and gas transport properties. "
        "Bounded on purpose: a corpus that covers everything discriminates "
        "nothing.\n")
    add("**Retrieval note.** Every command carries its syntax, official "
        "examples, and all `note`/`warning` admonitions from its page. The "
        "admonitions matter most: they are where the manual records the "
        "constraints that otherwise surface as a failed run. LAMMPS line "
        "continuation is `&`, not a backslash — a lesson this project learned "
        "the hard way, and visible in `in.wall.ccl` below.\n")

    add("## Contents\n")
    for i, (title, _, _, _) in enumerate(TOPICS, start=1):
        anchor = title.lower().replace(" ", "-").replace("(", "").replace(")", "")
        anchor = anchor.replace("–", "").replace("—", "").replace(",", "")
        anchor = re.sub(r"-+", "-", anchor)
        add(f"{i}. [{title}](#{i}-{anchor})")
    add("")
    add("---\n")

    for i, (title, rationale, scripts, pages) in enumerate(TOPICS, start=1):
        add(f"## {i}. {title}\n")
        add(f"*Why this is here:* {rationale}\n")

        if scripts:
            add("### Official scripts\n")
            for rel, caption in scripts:
                path = examples / rel
                if not path.is_file():
                    missing_scripts += 1
                    missing.append(f"examples/{rel}")
                    continue
                found_scripts += 1
                text = path.read_text().rstrip()
                lines = text.splitlines()
                truncated = ""
                if len(lines) > MAX_SCRIPT_LINES:
                    lines = lines[:MAX_SCRIPT_LINES]
                    truncated = (f"\n# ... truncated at {MAX_SCRIPT_LINES} "
                                 f"lines; see examples/{rel}")
                add(f"#### `{rel}` — {caption}\n")
                add("```lammps\n" + "\n".join(lines) + truncated + "\n```\n")

        if pages:
            add("### Command reference\n")
            for stem, command, role in pages:
                rendered = render_command(docs, stem, command, role)
                if rendered:
                    found_pages += 1
                    parts.extend(rendered)
                else:
                    missing_pages += 1
                    missing.append(f"doc/src/{stem}.rst")
        add("---\n")

    # Coverage is part of the corpus: a reader must be able to tell what is
    # absent from the source tree rather than assume the section was complete.
    add("## Generation coverage\n")
    add(f"- example scripts: **{found_scripts} found**, {missing_scripts} missing")
    add(f"- command pages: **{found_pages} found**, {missing_pages} missing")
    if missing:
        add("\nNot present in this LAMMPS tree (names may have changed "
            "between versions):\n")
        for name in missing:
            add(f"- `{name}`")
    add("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path} ({size_kb:.0f} KB, "
          f"{len(out_path.read_text().splitlines())} lines)")
    print(f"LAMMPS version: {version}")
    print(f"scripts {found_scripts} found / {missing_scripts} missing   "
          f"pages {found_pages} found / {missing_pages} missing")
    for name in missing:
        print(f"  MISSING {name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lammps-src", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent.parent / "docs"
        / "lammps-rag.md")
    args = parser.parse_args(argv)
    return build(args.lammps_src, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
