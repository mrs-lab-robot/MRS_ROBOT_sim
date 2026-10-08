# Runtime configurations

`versions.yaml` is the canonical compatibility lock for Python, Isaac Sim, Isaac Lab, and IsaacLab-Arena. Update the
Arena commit and its nested Isaac Lab source commit together, then rerun the matching container smoke test before
changing the supported baseline.

Commit portable defaults only. Machine-specific mount paths and secrets belong in ignored `configs/local/` files or
environment variables.
