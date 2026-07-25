Animus
======
A text to image Anima GUI for machines without CUDA, AVX2, and so on.

```
sudo apt-get install -y bash gir1.2-gtk-3.0 libcairo2-dev libgirepository1.0-dev libgtk-3-0 pkg-config python3-dev python3-gi python3-gi-cairo python3-pip python3-venv clang cmake ninja-build ccache git libopenblas-dev ; make install ; animus
```

Skip the source build with ```TORCH_PREBUILT=1 make install```.

Installs to ```$HOME/.local``` or set the ```PREFIX``` environment variable.
Run as ```animus```.
