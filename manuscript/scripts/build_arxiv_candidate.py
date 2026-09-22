"""Package the current manuscript dependencies without historical artifacts."""
from pathlib import Path
import hashlib
import json
import zipfile

root = Path(__file__).resolve().parents[1]
out = root / 'submission'
out.mkdir(exist_ok=True)
files = [root / name for name in ('main.tex', 'main.bbl', 'preamble.tex',
                                  'macros.tex', 'supplement.pdf', 'bib/references.bib')]
files += sorted((root / 'sections').glob('*.tex'))
files += [root / 'figures' / (name + '.pdf') for name in (
    'fig01_reference_correction', 'fig02_residual_design',
    'fig03_qm9_transfer', 'fig04_roughness_nmae')]
files += [root / 'si' / name for name in ('supplementary_information.tex',
                                         'corrected_content.tex', 'test_normalization_control.tex')]
files += sorted((root / 'si/tables/test_protocol_2026-09-22').glob('*.tex'))
files += sorted((root / 'si/figures/test_protocol_2026-09-22').glob('*.pdf'))
files += [root / 'si/build/supplementary_information.bbl']
manifest = {}
archive = out / 'Sep15_arxiv_source.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in files:
        assert p.is_file() and not p.is_symlink(), p
        name = str(p.relative_to(root))
        if name == 'si/build/supplementary_information.bbl':
            name = 'supplementary_information.bbl'
        z.write(p, name)
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    control = dict(process=dict(compiler='pdflatex'), sources=[
        dict(filename='main.tex', usage='toplevel'),
        dict(filename='si/supplementary_information.tex', usage='include')])
    z.writestr('00README.json', json.dumps(control, indent=2)+'\n')
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(archive)
print(f'{len(files)} files plus processing instructions; {archive.stat().st_size} bytes')
