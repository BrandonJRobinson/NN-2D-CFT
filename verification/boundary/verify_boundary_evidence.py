from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
errors=[]
for entry in manifest['files']:
    p=root/entry['path']
    if not p.is_file():
        errors.append(entry['path']+': missing');continue
    if p.stat().st_size!=entry['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        errors.append(entry['path']+': changed')
if errors:raise SystemExit('\n'.join(errors))
print('Verified '+str(len(manifest['files']))+' files.')
