"""Download the MIT teaching CSV for reproducible educational experiments."""
from pathlib import Path
from urllib.request import urlopen

url = "https://ocw.mit.edu/courses/15-071-the-analytics-edge-spring-2017/5d689a024551e672313f7fd7eb1bee8d_framingham.csv"
target = Path("data/raw/framingham.csv")
if target.exists():
    raise SystemExit("File already exists; refusing to replace it.")
target.parent.mkdir(parents=True, exist_ok=True)
with urlopen(url, timeout=60) as response:
    content = response.read()
if not content.startswith(b"male,age,") or b"TenYearCHD" not in content[:300]:
    raise SystemExit("Unexpected download schema; no file written.")
target.write_bytes(content)
print(f"Saved {target}. Teaching-source data: not cleared for clinical or publication claims.")
