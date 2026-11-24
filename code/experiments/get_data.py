"""Download the Bischoff-Ratcliff instances and check them against the files used in the paper.

Usage, from the code folder: python -m experiments.get_data

Classes BR1-BR7 come from the OR-Library of J. E. Beasley, classes BR8-BR15 from the OR-Datasets
collection of O. Oliveira, which holds all 15 classes in the same format. The files are not part of
this repository; they are fetched from these two sources and compared with the SHA-256 checksums of
the files the experiments were run on.
"""
import hashlib
import urllib.request
import zipfile

from .common import DATA, LARGE

ORLIB = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/thpack{}.txt"
ZIP = "https://raw.githubusercontent.com/Oscar-Oliveira/OR-Datasets/master/Cutting-and-Packing/3D/Datasets/BR/original.zip"
SHA256 = {
    "thpack1.txt": "d6f0b99b4cc6e51c5753565b175a778a40422dc92cffee3204bf16cc1d23f341",
    "thpack2.txt": "a31e6400d46926574068524172b1e38495cfd5ad6725f92c58a0af8d92721700",
    "thpack3.txt": "3359b278aa20a6874d5a63467be6064e8d489014df45f5410a7d85a852ef876d",
    "thpack4.txt": "9668bae0abd2ad513c6f8bcb5ed8845c4526f08b452f1b7fa9426b6990fbc4d6",
    "thpack5.txt": "5ece29990ab5034d75728c7ee777379475b16b9218fa9dded17f327de7fa126f",
    "thpack6.txt": "5c3604cad7a47f7ee38c3c6a4e74c0e846db1eea9ca4ac637106843311abe196",
    "thpack7.txt": "43fc06212473bbb12ec575678232e444e021642cec7b649be1d43282edec7f2a",
    "original.zip": "5cc9fa6d48a01ccf840f450f93887ede13b11d68b5dd821af23326219887eab3",
    "BR8.txt": "5c9824780eca22aab9298e2ef45c5d181e7a7a96129943addc79a0118287902b",
    "BR9.txt": "49926c26aff96b2dcfe3f98c162edb1f3666de41ed3e043718c866d03b55eeca",
    "BR10.txt": "8097a2f5d82470f7c5d7f196e16447f140048dd51f5a7c22dbc282ddfe759b02",
    "BR11.txt": "7bd31f975a0460ca1051e4e28f649fdb42eac2fb0c732869e03fdfa81241ce04",
    "BR12.txt": "c7d0e32666c176b015b77e27e0d555ce6d5d1a88691bd869b53b78c106dda54d",
    "BR13.txt": "d2a454cf60198d4ed1acf6b9a613f6f9f94f610bb3bc7e00e1c52f84528bc121",
    "BR14.txt": "bd036b3ddabaa47a4692c67419717e5980d1684f9a1f8b0ed6c9d50fb92bb90f",
    "BR15.txt": "598dc5d805000c856b2c802296633a11ff90e7bdb849c72d00212272cc3a2f6f",
}


def fetch(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    check(path)


def check(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    status = "ok" if digest == SHA256[path.name] else "DIFFERENT FROM THE FILE USED IN THE PAPER"
    print(f"{path.name}: {status}")


if __name__ == "__main__":
    for k in range(1, 8):
        fetch(ORLIB.format(k), DATA / f"thpack{k}.txt")
    archive = LARGE.parent / "original.zip"
    fetch(ZIP, archive)
    with zipfile.ZipFile(archive) as z:
        for k in range(8, 16):
            name = next(n for n in z.namelist() if n.endswith(f"BR{k}.txt"))
            target = LARGE / f"BR{k}.txt"
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
            check(target)
