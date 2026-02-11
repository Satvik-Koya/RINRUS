"""Utilities for converting PyMOL-exported PDB structures to Psi4 input files."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Set

from atom_element import Atoms


DEFAULT_IONS = {
    "LI", "NA", "K", "RB", "CS", "BE", "MG", "CA", "SR", "BA",
    "AL", "GA", "ZN", "CD", "HG", "FE", "CO", "NI", "CU", "MN",
    "CR", "V", "MO", "W", "AG", "AU", "PB", "SN", "TL", "U", "PT",
    "CL", "BR", "I", "F",
}

WATER_RESNAMES = {"HOH", "WAT", "H2O"}
TWO_LETTER_ELEMENTS = {
    "HE", "LI", "BE", "NE", "NA", "MG", "AL", "SI", "CL", "AR", "CA",
    "SC", "TI", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE",
    "AS", "SE", "BR", "KR", "RB", "SR", "ZR", "NB", "MO", "TC", "RU",
    "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "XE", "CS", "BA",
    "LA", "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO",
    "ER", "TM", "YB", "LU", "HF", "TA", "RE", "OS", "IR", "PT", "AU",
    "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM", "MD", "NO",
    "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN", "NH",
    "FL", "MC", "LV", "TS", "OG",
}
DFT_METHOD_HINTS = {
    "b3lyp", "pbe", "pbe0", "wb97", "m06", "tpss", "b97", "blyp", "bp86",
    "scan", "revpbe", "hse", "pw6b95", "mn15", "b2plyp",
}


@dataclass
class PDBAtom:
    serial: int
    atom_name: str
    altloc: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str


def normalize_element(raw: str) -> Optional[str]:
    token = raw.strip()
    if not token:
        return None
    token = token[0].upper() + token[1:].lower()
    return token if token in Atoms else None


def infer_element(atom_name: str, element_column: str = "") -> str:
    explicit = normalize_element(element_column)
    if explicit:
        return explicit

    if len(atom_name) < 4:
        atom_name = atom_name.ljust(4)

    # PDB formatting rule: two-letter elements are typically left-justified.
    if atom_name[0].isalpha():
        first_two = atom_name[:2].strip().upper()
        if len(first_two) == 2 and first_two in TWO_LETTER_ELEMENTS:
            guess = first_two[0].upper() + first_two[1].lower()
            if guess in Atoms:
                return guess
        first = atom_name[0].upper()
        if first in Atoms:
            return first

    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        raise ValueError(f"Unable to infer element from atom name '{atom_name}'")

    # Safeguard: atom name ' CA ' in proteins should resolve to carbon.
    first = letters[0].upper()
    if first in Atoms:
        return first

    raise ValueError(f"Unable to infer element from atom name '{atom_name}'")


def parse_pdb_atoms(
    pdb_path: str,
    allowed_altlocs: Optional[Set[str]] = None,
    exclude_waters: bool = False,
    exclude_ions: bool = False,
    ion_resnames: Optional[Set[str]] = None,
) -> List[PDBAtom]:
    allowed = {"", "A"} if allowed_altlocs is None else {a.strip().upper() for a in allowed_altlocs}
    ions = DEFAULT_IONS if ion_resnames is None else {i.strip().upper() for i in ion_resnames}
    atoms: List[PDBAtom] = []

    with open(pdb_path, "r") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if len(line) < 54:
                raise ValueError(f"Malformed PDB line {lineno}: coordinates are incomplete")

            atom_name = line[12:16]
            altloc = line[16].strip().upper()
            resname = line[17:20].strip().upper()

            if altloc not in allowed:
                continue
            if exclude_waters and resname in WATER_RESNAMES:
                continue

            serial = int(line[6:11])
            resseq = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element_column = line[76:78] if len(line) >= 78 else ""
            element = infer_element(atom_name, element_column)

            if exclude_ions:
                if resname in ions or (resname == element.upper() and element.upper() in ions):
                    continue

            atoms.append(
                PDBAtom(
                    serial=serial,
                    atom_name=atom_name,
                    altloc=altloc,
                    resname=resname,
                    chain=line[21].strip(),
                    resseq=resseq,
                    x=x,
                    y=y,
                    z=z,
                    element=element,
                )
            )

    if not atoms:
        raise ValueError("No atoms were selected from the PDB file after filtering")
    return atoms


def parse_index_ranges(spec: str) -> List[int]:
    if not spec.strip():
        return []
    out: Set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2 or not bounds[0].isdigit() or not bounds[1].isdigit():
                raise ValueError(f"Invalid range token '{token}'")
            start, end = int(bounds[0]), int(bounds[1])
            if start > end:
                raise ValueError(f"Invalid range token '{token}': start exceeds end")
            out.update(range(start, end + 1))
        else:
            if not token.isdigit():
                raise ValueError(f"Invalid freeze token '{token}'")
            out.add(int(token))
    return sorted(out)


def map_frozen_atoms(atoms: Sequence[PDBAtom], freeze_mode: str, freeze_spec: str) -> List[int]:
    if freeze_mode == "none":
        if freeze_spec.strip():
            raise ValueError("--freeze requires --freeze-mode pdb-serial or atom-index")
        return []

    values = parse_index_ranges(freeze_spec)
    if not values:
        return []

    if freeze_mode == "atom-index":
        max_idx = len(atoms)
        bad = [i for i in values if i < 1 or i > max_idx]
        if bad:
            raise ValueError(f"Atom indices out of range: {bad}; valid range is 1..{max_idx}")
        return values

    if freeze_mode == "pdb-serial":
        serial_to_index = {atom.serial: idx + 1 for idx, atom in enumerate(atoms)}
        mapped = []
        missing = []
        for serial in values:
            idx = serial_to_index.get(serial)
            if idx is None:
                missing.append(serial)
            else:
                mapped.append(idx)
        if missing:
            raise ValueError(f"PDB serial(s) not present after filtering: {missing}")
        return sorted(set(mapped))

    raise ValueError(f"Unsupported freeze mode '{freeze_mode}'")


def method_is_dft(method: str, is_dft_override: Optional[bool] = None) -> bool:
    if is_dft_override is not None:
        return is_dft_override
    token = method.lower()
    return any(hint in token for hint in DFT_METHOD_HINTS)


def build_psi4_input(
    atoms: Sequence[PDBAtom],
    source_filename: str,
    output_basename: str,
    charge: int,
    multiplicity: int,
    method: str = "b3lyp-d3bj",
    basis: str = "def2-svp",
    job: str = "energy",
    mem_gb: int = 8,
    threads: int = 8,
    freeze_indices: Optional[Sequence[int]] = None,
    is_dft_override: Optional[bool] = None,
    timestamp: Optional[str] = None,
) -> str:
    if job not in {"energy", "opt", "freq"}:
        raise ValueError("job must be one of: energy, opt, freq")

    if timestamp is None:
        timestamp = datetime.now().isoformat(timespec="seconds")

    lines = [
        f"# generated by RINRUS PyMOL->Psi4 converter",
        f"# source: {source_filename}",
        f"# generated: {timestamp}",
        "",
        f"memory {mem_gb} GB",
        f"set_num_threads({threads})",
        f"set_output_file(\"{output_basename}.out\", False)",
        "",
        "molecule mol {",
        f"  {charge} {multiplicity}",
        "  units angstrom",
        "  symmetry c1",
        "  no_com",
        "  no_reorient",
    ]

    for atom in atoms:
        lines.append(f"  {atom.element:<2} {atom.x: .8f} {atom.y: .8f} {atom.z: .8f}")
    lines.extend(["}", "", "set {"])
    lines.extend([
        f"  basis {basis}",
        "  scf_type df",
        "  guess sad",
        "  maxiter 200",
        "  e_convergence 1e-8",
        "  d_convergence 1e-8",
        "  soscf true",
    ])
    if method_is_dft(method, is_dft_override=is_dft_override):
        lines.extend([
            "  dft_radial_points 99",
            "  dft_spherical_points 590",
        ])
    lines.extend(["}", ""])

    if job == "opt":
        lines.append("set optking {")
        lines.append("  g_convergence gau_tight")
        if freeze_indices:
            constraints = ", ".join(f"{idx} xyz" for idx in freeze_indices)
            lines.append(f"  frozen_cartesian [{constraints}]")
        lines.append("}")
        lines.append("")

    call_map = {
        "energy": "energy",
        "opt": "optimize",
        "freq": "frequency",
    }
    lines.append(f"{call_map[job]}('{method}')")
    lines.append("")
    return "\n".join(lines)


def convert_pymol_pdb_to_psi4(
    pdb_path: str,
    output_path: str,
    charge: int,
    multiplicity: int,
    method: str = "b3lyp-d3bj",
    basis: str = "def2-svp",
    job: str = "energy",
    mem_gb: int = 8,
    threads: int = 8,
    exclude_waters: bool = False,
    exclude_ions: bool = False,
    allowed_altlocs: Optional[Set[str]] = None,
    ion_resnames: Optional[Set[str]] = None,
    freeze_mode: str = "none",
    freeze_spec: str = "",
    is_dft_override: Optional[bool] = None,
) -> str:
    atoms = parse_pdb_atoms(
        pdb_path,
        allowed_altlocs=allowed_altlocs,
        exclude_waters=exclude_waters,
        exclude_ions=exclude_ions,
        ion_resnames=ion_resnames,
    )
    freeze_indices = map_frozen_atoms(atoms, freeze_mode=freeze_mode, freeze_spec=freeze_spec)
    basename = Path(output_path).with_suffix("").name
    text = build_psi4_input(
        atoms=atoms,
        source_filename=Path(pdb_path).name,
        output_basename=basename,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        basis=basis,
        job=job,
        mem_gb=mem_gb,
        threads=threads,
        freeze_indices=freeze_indices,
        is_dft_override=is_dft_override,
    )
    with open(output_path, "w") as handle:
        handle.write(text)
    return text
