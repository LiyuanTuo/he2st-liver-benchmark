"""Check the prospective Git payload without staging or publishing anything."""
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    raw = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'], cwd=ROOT)
    paths = sorted(set(p.decode('utf8') for p in raw.split(b'\0') if p))
    forbidden = {'.npz', '.npy', '.h5ad', '.h5', '.pt', '.pth', '.ckpt', '.tif', '.tiff', '.pdf', '.pptx', '.gz', '.zip'}
    curated = {'report/main.pdf', 'presentation/final_report.pptx', 'presentation/final_report.pdf'}
    curated.update(p for p in paths if p.startswith('report/figures/') and Path(p).suffix=='.pdf')
    violations = [p for p in paths if (Path(p).suffix.lower() in forbidden and p not in curated)
                  or (ROOT/p).stat().st_size > (20_000_000 if p=='presentation/final_report.pptx' else 5_000_000)]
    for path in paths:
        if path.startswith(('third_party/', 'data/raw/', 'data/processed/', 'results/')) and path != 'results/README.md':
            violations.append(path)
        if Path(path).name in ['.env', 'requirement.txt']:
            violations.append(path)
    result = {'files': len(paths), 'bytes': sum((ROOT/p).stat().st_size for p in paths),
              'forbidden_or_large_files': sorted(set(violations)), 'staged_or_published_by_audit': False}
    print(json.dumps(result, indent=2))
    if violations:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
