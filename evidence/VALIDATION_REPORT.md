# Validation Report

**Validation date:** 3 August 2026  
**Scope:** Bundled offline bootstrap artifact and repository integrity.  
**Important:** Bootstrap metrics use deterministic generated images, not the Kaggle generalization dataset.

## Completed checks

| Check | Result | Evidence |
|---|---|---|
| Python source compilation | PASS | `python -m compileall -q src tests` |
| Unit/API tests | PASS — 4 tests | `evidence/pytest_output.txt` |
| Model deserialization | PASS | FastAPI startup log and `/health` |
| Health endpoint | PASS | `evidence/smoke_test_output.json` |
| Prediction endpoint | PASS | Normalized cat/dog probabilities and request ID in smoke output |
| Delayed-label feedback | PASS | Eight labeled requests recorded |
| Online performance calculation | PASS | `evidence/post_deploy_batch_output.json` |
| Prometheus exposition | PASS | `evidence/prometheus_metrics.txt` |
| YAML syntax | PASS | CI/CD, Compose, DVC, parameters, Kubernetes and monitoring files parsed |
| Report rendering | PASS | DOCX rendered to 11-page PDF and visually inspected |

## Bootstrap runtime result

- Model architecture: TinyCNN
- Input: 224x224 RGB
- Smoke-test label: cat
- Smoke-test confidence: 0.767061
- Labeled post-deployment requests: 8
- Correct labeled requests: 8
- Bootstrap online accuracy: 1.0

The value 1.0 is expected for the deliberately simple generated bootstrap set and must not be reported as Kaggle test accuracy. Execute the Kaggle/DVC procedure in `RUNBOOK.md` to generate final full-dataset evidence.

## Environment-dependent checks

Docker image build, GHCR push, GitHub-hosted CI/CD, Kaggle download, and persistent remote deployment require external services/account credentials. Their complete executable configuration is included, but no credentials are embedded in the archive.

Repository validation evidence added on 2026-08-03.
