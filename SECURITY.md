# Security and privacy controls

- Image payloads are never written to application logs.
- Request logs contain only request ID, route, method, status, and latency.
- Upload size is limited to 10 MB and only image content types are accepted.
- The container runs as a non-root user.
- Secrets such as Kaggle credentials, registry tokens, and SSH keys must be stored in local secret files or GitHub Actions secrets, never committed.
- CI creates SBOM and provenance attestations for the container image.
- Replace default Grafana credentials before any non-local deployment.
