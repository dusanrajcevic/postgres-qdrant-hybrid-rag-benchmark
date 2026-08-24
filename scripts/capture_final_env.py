from __future__ import annotations
import json, platform, subprocess, sys
from pathlib import Path
from common import ROOT, load_config, write_json

OUT = ROOT / "results" / "measurements" / "100000" / "environment_snapshot.json"

def run(cmd):
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, check=False)
        return {
            "command": cmd,
            "returncode": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except Exception as e:
        return {"command": cmd, "error": repr(e)}

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "docker_version": run(["docker","version","--format","{{json .}}"]),
        "docker_compose_version": run(["docker","compose","version"]),
        "docker_compose_ps": run(["docker","compose","ps"]),
        "postgres_versions": run([
            "docker","compose","exec","-T","postgres","psql",
            "-U","benchmark","-d","benchmark","-Atc",
            "SELECT version(); SELECT extname||'='||extversion FROM pg_extension "
            "WHERE extname IN ('vector','vectorscale') ORDER BY extname;"
        ]),
        "qdrant_image": run([
            "docker","compose","images","qdrant"
        ]),
        "postgres_image": run([
            "docker","compose","images","postgres"
        ]),
        "compose_config": run(["docker","compose","config"]),
    }
    write_json(OUT, data)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
