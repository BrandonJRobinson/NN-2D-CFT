from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
errors=[]
for entry in manifest['files']:
    p=root/entry['path']
    if not p.is_file() or p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        errors.append(entry['path'])
if errors:raise SystemExit('Missing or changed files: '+', '.join(errors))
print('Verified '+str(len(manifest['files']))+' files.')
