#!/usr/bin/env python3
"""生成 version.json：供 Mac 包内页面显示版本号 + 双击查看更新日志。

用法：
    python3 tools/gen-version.py [输出路径] [分支] [条数]
默认：/tmp/nn-anatomy2-version.json，分支 build，取最近 30 条提交。

打包流程（tools/serve.command 同级）：
    git archive --format=zip -o pkg.zip build
    python3 tools/gen-version.py /tmp/version.json
    python3 -c "import zipfile; z=zipfile.ZipFile('pkg.zip','a'); z.write('/tmp/version.json','version.json'); z.close()"
"""
import json
import subprocess
import sys


def run(*args):
    return subprocess.run(
        args, capture_output=True, text=True, check=True
    ).stdout.strip()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nn-anatomy2-version.json"
    branch = sys.argv[2] if len(sys.argv) > 2 else "build"
    count = sys.argv[3] if len(sys.argv) > 3 else "30"

    log = run(
        "git", "log", branch,
        "--date=format:%Y-%m-%d %H:%M",
        "--pretty=format:%H|%h|%ad|%s",
        f"-{count}",
    )
    entries = []
    for line in log.splitlines():
        full, short, date, subject = line.split("|", 3)
        entries.append({"hash": full, "short": short, "date": date, "subject": subject})

    head = entries[0] if entries else {"hash": "", "short": "unknown", "date": "", "subject": ""}
    payload = {
        "version": f"{head['date'][:10]}-{head['short']}" if head["date"] else "unknown",
        "built_from": {"hash": head["hash"], "short": head["short"], "date": head["date"], "subject": head["subject"]},
        "changelog": entries,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out}: version={payload['version']} entries={len(entries)}")


if __name__ == "__main__":
    main()
