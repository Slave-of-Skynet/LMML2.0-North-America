"""Package only main.py and the data-only index/ directory for submission."""
import argparse
import hashlib
from pathlib import Path
import zipfile

FORBIDDEN = {".py", ".pyc", ".pyo", ".so", ".pyd", ".dll", ".dylib", ".exe", ".sh", ".bat", ".cmd", ".ps1"}


def package(index_dir, destination):
    index_dir, destination = Path(index_dir), Path(destination)
    if destination.resolve().is_relative_to(index_dir.resolve()):
        raise ValueError('The ZIP must be outside index/')
    files = sorted(index_dir.rglob("*"))
    for path in files:
        if path.is_symlink() or path.suffix.lower() in FORBIDDEN:
            raise ValueError("Forbidden index entry: " + str(path))
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = [(Path(__file__).with_name('main.py'), 'main.py')]
    entries.extend((path, 'index/' + path.relative_to(index_dir).as_posix()) for path in files if path.is_file())
    size = sum(path.stat().st_size for path, _ in entries)
    if size > 53_000_000:
        raise ValueError('Unpacked submission exceeds 53 decimal MB: ' + str(size))
    required = {'config.json', 'manifest.json', 'lexicon.json.gz', 'chunk_doc.npy', 'ptr.npy', 'ids.npy', 'values.npy'}
    if not all((index_dir/name).is_file() for name in required):
        raise ValueError('Index is incomplete')
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in entries:
            info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=6)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Archive CRC check failed")
        print("Files:", len(archive.namelist()), "unpacked bytes:", sum(i.file_size for i in archive.infolist()))
    print(destination.resolve(), destination.stat().st_size, "compressed bytes")
    print('SHA-256:', hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dir")
    parser.add_argument("output_zip")
    args = parser.parse_args()
    package(args.index_dir, args.output_zip)
