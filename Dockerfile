ARG CUDA_VERSION=13.0.3
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS build

# llama.cpp release with /v1/systemone (ggml-org/llama.cpp#29818)
ARG LLAMA_CPP_VERSION=b11559
# GB10 (Blackwell, DGX Spark / ASUS Ascent GX10)
ARG CUDA_ARCH=121

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${LLAMA_CPP_VERSION} https://github.com/ggml-org/llama.cpp /src

WORKDIR /src

RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    && cmake --build build --config Release -j"$(nproc)" --target llama-server llama-quantize

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libssl3 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/build/bin/llama-server /src/build/bin/llama-quantize /app/

WORKDIR /app

ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

EXPOSE 8080

HEALTHCHECK CMD ["curl", "-f", "http://localhost:8080/health"]

ENTRYPOINT ["/app/llama-server"]
