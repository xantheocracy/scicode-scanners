# Sandbox chart provenance

Vendored from the official inspect-k8s-sandbox 0.6.1 wheel (agent-env chart 0.12.0).

Local change: add automountServiceAccountToken to the values schema and pod template, defaulting to false. The eval values explicitly disable token mounts for both investigation and grader services.
