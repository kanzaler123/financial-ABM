"""Download the MOF China government bond yield curve, one file per year (contract section 26.6).

    python -I -X utf8 download_curve.py

Source: 财政部-中国国债收益率曲线历史数据, compiled by China Central Depository & Clearing and published at
https://yield.chinabond.com.cn/cbweb-mn/pgxh/showHistory (tenors 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 30Y; a query
may not span a year or more). The page's own download button calls historyDownload with gjqx=0 (all tenors).
Files go to data/raw/chinabond_mof_curve_20261027/ (not committed; the site asks for attribution when data are
reproduced). curve_manifest.json here records each URL, the server's file name, size and SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from email.message import Message
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data/raw/chinabond_mof_curve_20261027"
URL = "https://yield.chinabond.com.cn/cbweb-mn/pgxh/historyDownload?startDate={y}-01-01&&endDate={y}-12-31&&gjqx=0"
YEARS = range(2009, 2026)


def server_name(header: str | None) -> str | None:
    if not header:
        return None
    msg = Message()
    msg["content-disposition"] = header
    name = msg.get_param("filename", header="content-disposition")
    if name is None:
        return None
    try:
        return name.encode("latin-1").decode("gbk")        # the server sends the GBK bytes of the Chinese name
    except (UnicodeEncodeError, UnicodeDecodeError):
        return urllib.parse.unquote(name)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = []
    for y in YEARS:
        url = URL.format(y=y)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research download; contract 26)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            body, disposition, ctype = r.read(), r.headers.get("Content-Disposition"), r.headers.get("Content-Type")
        assert body[:2] == b"PK", f"{y}: not an xlsx file ({ctype})"
        path = OUT / f"mof_cgb_curve_{y}.xlsx"
        path.write_bytes(body)
        files.append({"year": y, "url": url, "server_file_name": server_name(disposition), "file": path.name,
                      "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
        print(y, len(body))
        time.sleep(1.0)
    manifest = {"source": "财政部-中国国债收益率曲线 (MOF China government bond yield curve), compiled by CCDC",
                "page": "https://yield.chinabond.com.cn/cbweb-mn/pgxh/showHistory",
                "downloaded": time.strftime("%Y-%m-%d %H:%M:%S %z"), "directory": OUT.relative_to(ROOT).as_posix(),
                "files": files}
    (HERE / "curve_manifest.json").write_bytes((json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()
