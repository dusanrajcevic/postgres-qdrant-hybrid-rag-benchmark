# Original benchmark environment

The final measurements were collected on a MacBook Pro with Apple M1 Pro and 16 GB unified memory. The databases ran as native ARM64 Docker containers.

| Component | Version or setting |
| --- | --- |
| macOS | 26.6.2, build 25G83 |
| Host CPU | Apple M1 Pro, 10 physical/logical CPUs reported by macOS |
| Host memory | 16 GB |
| Docker Desktop | 4.87.0 |
| Docker Engine | 29.7.2 |
| Docker Compose | 5.4.0 |
| Container architecture | linux/arm64 |
| CPU limit per database service | 6 CPUs |
| Memory limit per database service | 8 GB |
| PostgreSQL shared memory | 1 GB |
| Python | 3.11.11 |
| PostgreSQL | 18.4 |
| pgvector | 0.8.6 |
| pgvectorscale | 0.9.0 |
| Qdrant | 1.18.1 |

Container images used by the recorded run:

- `timescale/timescaledb-ha:pg18.4-ts2.29.1-all-oss`
- `qdrant/qdrant:v1.18.1`

Image digests captured during the experiment:

- Timescale/PostgreSQL: `sha256:9bfff27f88e7a78e3e6d6de66d20b46200015a9db79082ce13ccd7316a26e1c0`
- Qdrant: `sha256:45f8e3ddc2570a4d029877e1b5ec1045c19b3852b4e22a55c7f43b05aea0ca89`

The original environment snapshots used for the final runs are retained in each measurement directory. They include the resolved Compose configuration and database version checks without the host-specific Docker plugin paths that were present in the development environment log.

Python dependency files record the compatible dependency ranges used by the scripts. The individual Python wheel versions were not captured as a complete `pip freeze` during the original measurements, so this repository does not claim an exact Python package lock that was not recorded at experiment time.
