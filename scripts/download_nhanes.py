"""Download official CDC baseline and linked mortality files (about 320 MB)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

ROOT = Path("data/raw/nhanes3")
BASE = "https://wwwn.cdc.gov/nchs/data/nhanes3/"
MORT = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
FILES = {f"{name}.{ext}": BASE + f"{release}/{name}.{ext}"
         for name, release in [("adult", "1a"), ("exam", "1a"), ("lab", "1a"), ("nh3ecg", "2a")]
         for ext in ["sas", "dat"]}
FILES.update({"mortality.dat": MORT + "NHANES_III_MORT_2019_PUBLIC.dat",
              "mortality.sas": MORT + "SAS_ReadInProgramAllSurveys.sas"})


def download(item):
    name, url = item
    path = ROOT / name
    if path.exists():
        return f"Exists: {name}"
    temp = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=120) as response, temp.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
    if temp.read_bytes()[:100].lstrip().startswith(b"<"):
        raise ValueError(f"Unexpected HTML instead of {name}")
    temp.rename(path)
    return f"Downloaded {name}: {path.stat().st_size} bytes"


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(download, FILES.items()):
            print(result, flush=True)
