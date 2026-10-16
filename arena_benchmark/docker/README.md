# Container Support

This repository does not currently provide or maintain a project-specific container image. The upstream simulator/Arena container may be used as a development starting point, but this repository has not established a clean-container deployment or end-to-end acceptance path.

Any container setup must use the version lock in [`../../configs/versions.yaml`](../../configs/versions.yaml), mount the repository assets at a path visible inside the container, and follow NVIDIA/Omniverse license requirements. Do not treat an upstream image as a validated release image for this project.