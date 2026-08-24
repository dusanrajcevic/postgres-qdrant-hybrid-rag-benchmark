from __future__ import annotations
import platform
import subprocess
import sys

from common import ROOT, write_json

OUT = ROOT / "results" / "measurements" / "250000" / "environment_snapshot.json"

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
        "docker_version": run(["docker", "version", "--format", "{{json .}}"]),
        "docker_compose_version": run(["docker", "compose", "version"]),
        "docker_compose_ps": run(["docker", "compose", "ps"]),
        "postgres_versions": run([
            "docker", "compose", "exec", "-T", "postgres", "psql",
            "-U", "benchmark", "-d", "benchmark", "-Atc",
            "SELECT version(); SELECT extname||'='||extversion FROM pg_extension "
            "WHERE extname IN ('vector','vectorscale') ORDER BY extname;"
        ]),
        "postgres_image": run(["docker", "compose", "images", "postgres"]),
        "qdrant_image": run(["docker", "compose", "images", "qdrant"]),
        "compose_config": run(["docker", "compose", "config"]),
    }
    write_json(OUT, data)
    print(f"Wrote {OUT}")

if __name__ == "__main__":
    main()
