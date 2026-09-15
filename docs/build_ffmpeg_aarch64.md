# Build ffmpeg (aarch64, no sudo)

This installs ffmpeg in user space so MP3 decoding works for the audio
tokenizer benchmark.

## Build libsoxr (high-quality resampler)

```bash
PREFIX=$HOME/.local/ffmpeg
mkdir -p "$PREFIX"

cd /tmp
wget https://sourceforge.net/projects/soxr/files/soxr-0.1.3-Source.tar.xz
tar -xf soxr-0.1.3-Source.tar.xz
cd soxr-0.1.3-Source

cmake -B build \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DWITH_OPENMP=ON

cmake --build build -j"$(nproc)"
cmake --install build
```

## Build ffmpeg with soxr

```bash
cd /tmp
wget https://ffmpeg.org/releases/ffmpeg-6.1.1.tar.xz
tar -xf ffmpeg-6.1.1.tar.xz
cd ffmpeg-6.1.1

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"

./configure --prefix="$PREFIX" \
  --enable-gpl \
  --enable-libsoxr \
  --disable-debug \
  --disable-doc \
  --disable-x86asm \
  --extra-cflags="-I$PREFIX/include" \
  --extra-ldflags="-L$PREFIX/lib"

make -j"$(nproc)"
make install
```

## Use in shell or job script

```bash
export PATH="$PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
ffmpeg -version | head -n 1
```

## Verify soxr support

```bash
ffmpeg -hide_banner -h filter=aresample 2>&1 | grep -q soxr && echo "soxr: OK" || echo "soxr: MISSING"
ldd $(which ffmpeg) | grep soxr
```

## Notes

- This is for aarch64; `--disable-x86asm` avoids x86-only assembly.
- The install location is `$HOME/.local/ffmpeg`, which is typically shared
  across compute nodes.
- libsoxr provides high-quality resampling (better than ffmpeg's default).
