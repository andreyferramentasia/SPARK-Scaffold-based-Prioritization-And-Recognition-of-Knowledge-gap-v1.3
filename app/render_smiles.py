"""CLI helper: render a SMILES string to a PNG image using RDKit.

Usage:
    python render_smiles.py <smiles_file> <output.png> [width] [height]

smiles_file: plain-text file containing the SMILES (avoids shell-escaping issues).
Exit codes: 0=ok, 1=bad args/file, 2=rdkit not installed, 3=invalid SMILES.
"""
import sys
import re


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: render_smiles.py <smiles_file> <output.png> [width] [height]",
              file=sys.stderr)
        sys.exit(1)

    smiles_file = sys.argv[1]
    output_path = sys.argv[2]
    width  = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    height = int(sys.argv[4]) if len(sys.argv) > 4 else 400

    try:
        with open(smiles_file, encoding="utf-8") as fh:
            smiles = fh.read().strip()
    except Exception as exc:
        print(f"Cannot read SMILES file: {exc}", file=sys.stderr)
        sys.exit(1)

    if not smiles:
        print("Empty SMILES", file=sys.stderr)
        sys.exit(1)

    try:
        from rdkit import Chem
        from rdkit.Chem import Draw, rdDepictor
    except ImportError:
        import subprocess
        print("rdkit not found — installing automatically...", file=sys.stderr, flush=True)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "rdkit"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"Failed to install rdkit:\n{result.stderr}", file=sys.stderr)
            sys.exit(2)
        try:
            from rdkit import Chem
            from rdkit.Chem import Draw, rdDepictor
        except ImportError:
            print("rdkit install succeeded but import still failed.", file=sys.stderr)
            sys.exit(2)

    try:
        rdDepictor.SetPreferCoordGen(True)
    except Exception:
        pass

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # strip stereo / directional bonds and retry
        smiles_clean = re.sub(r'[@/\\]', '', smiles)
        mol = Chem.MolFromSmiles(smiles_clean)

    if mol is None:
        print(f"Cannot parse SMILES: {smiles[:80]}", file=sys.stderr)
        sys.exit(3)

    img = Draw.MolToImage(mol, size=(width, height))
    img.save(output_path)
    sys.exit(0)


if __name__ == "__main__":
    main()
