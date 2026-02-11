import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib3"))

from pymol_psi4 import parse_pdb_atoms, map_frozen_atoms, build_psi4_input


def pdb_line(record, serial, atom, altloc, res, chain, resseq, x, y, z, element=""):
    return f"{record:<6}{serial:5d} {atom:<4}{altloc:1}{res:>3} {chain:1}{resseq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00          {element:>2}\n"


class TestPyMOLPsi4Converter(unittest.TestCase):
    def write_tmp_pdb(self, contents):
        fh = tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False)
        fh.write(contents)
        fh.close()
        self.addCleanup(lambda: os.path.exists(fh.name) and os.remove(fh.name))
        return fh.name

    def test_element_column_present(self):
        pdb = (
            pdb_line("ATOM", 1, "CA", " ", "ALA", "A", 1, 1.0, 2.0, 3.0, "C")
            + pdb_line("HETATM", 2, "ZN", " ", "ZN", "A", 2, 4.0, 5.0, 6.0, "ZN")
        )
        path = self.write_tmp_pdb(pdb)
        atoms = parse_pdb_atoms(path)
        self.assertEqual([a.element for a in atoms], ["C", "Zn"])

    def test_element_inference_ca_vs_calcium(self):
        # Alpha carbon with blank first column should become carbon; calcium should remain calcium.
        alpha_carbon = pdb_line("ATOM", 1, " CA ", " ", "ALA", "A", 1, 1, 1, 1, "")
        calcium = pdb_line("HETATM", 2, "CA  ", " ", "CA", "A", 2, 2, 2, 2, "")
        path = self.write_tmp_pdb(alpha_carbon + calcium)
        atoms = parse_pdb_atoms(path)
        self.assertEqual(atoms[0].element, "C")
        self.assertEqual(atoms[1].element, "Ca")

    def test_altloc_default_selects_blank_and_a(self):
        pdb = (
            pdb_line("ATOM", 1, "CA", "A", "ALA", "A", 1, 1.0, 2.0, 3.0, "C")
            + pdb_line("ATOM", 2, "CA", "B", "ALA", "A", 1, 1.1, 2.1, 3.1, "C")
            + pdb_line("ATOM", 3, "N", " ", "ALA", "A", 1, 0.0, 0.0, 0.0, "N")
        )
        path = self.write_tmp_pdb(pdb)
        atoms = parse_pdb_atoms(path)
        self.assertEqual([a.serial for a in atoms], [1, 3])

    def test_freeze_mapping_and_invalid_serial(self):
        pdb = (
            pdb_line("ATOM", 10, "N", " ", "ALA", "A", 1, 0, 0, 0, "N")
            + pdb_line("ATOM", 20, "CA", " ", "ALA", "A", 1, 1, 0, 0, "C")
            + pdb_line("ATOM", 30, "C", " ", "ALA", "A", 1, 2, 0, 0, "C")
        )
        path = self.write_tmp_pdb(pdb)
        atoms = parse_pdb_atoms(path)
        mapped = map_frozen_atoms(atoms, freeze_mode="pdb-serial", freeze_spec="10,30")
        self.assertEqual(mapped, [1, 3])
        with self.assertRaises(ValueError):
            map_frozen_atoms(atoms, freeze_mode="pdb-serial", freeze_spec="999")


    def test_custom_ion_filter_list(self):
        pdb = (
            pdb_line("HETATM", 1, "ZN", " ", "ZN", "A", 1, 0, 0, 0, "ZN")
            + pdb_line("HETATM", 2, "CU", " ", "CUA", "A", 2, 1, 0, 0, "CU")
            + pdb_line("ATOM", 3, "CA", " ", "ALA", "A", 3, 2, 0, 0, "C")
        )
        path = self.write_tmp_pdb(pdb)
        atoms = parse_pdb_atoms(path, exclude_ions=True, ion_resnames={"CUA"})
        self.assertEqual([a.serial for a in atoms], [1, 3])

    def test_opt_input_contains_constraints(self):
        pdb = pdb_line("ATOM", 1, "N", " ", "GLY", "A", 1, 0, 0, 0, "N")
        path = self.write_tmp_pdb(pdb)
        atoms = parse_pdb_atoms(path)
        text = build_psi4_input(
            atoms=atoms,
            source_filename="x.pdb",
            output_basename="input",
            charge=0,
            multiplicity=1,
            job="opt",
            freeze_indices=[1],
        )
        self.assertIn("set optking", text)
        self.assertIn("frozen_cartesian [1 xyz]", text)


if __name__ == "__main__":
    unittest.main()
