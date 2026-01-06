uv pip install draccus==0.11.5 matplotlib==3.10.1 ipython==8.38.0 msgpack==1.1.0 zmq==0.0.0

uv pip install --no-deps git+https://github.com/huggingface/lerobot.git@c75455a6de5c818fa1bb69fb2d92423e86c70475

uv pip uninstall opencv-python-headless
uv pip install opencv-python==4.11.0.86