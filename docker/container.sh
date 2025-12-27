# Interactive shell (uses code baked into image):
# sudo docker run -it --rm --gpus all gr00t-dev /bin/bash

# Development mode (mounts local codebase for live editing):
sudo docker run -it --rm --gpus all \
  --name gr00t \
  -v $(pwd)/..:/workspace/gr00t \
  -v /data:/data \
  gr00t-dev /bin/bash


# docker exec -it suspicious_noether /bin/bash    