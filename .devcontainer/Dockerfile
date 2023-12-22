FROM mcr.microsoft.com/devcontainers/cpp:0-ubuntu-22.04

#OS名を設定
ENV OS_NAME=Linux

RUN echo "ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true" | debconf-set-selections

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt install -y \
    build-essential \
    ninja-build \
    libeigen3-dev \
    libpython3-all-dev \
    python3-numpy \
    python3-matplotlib \
    python3-pip \
    msttcorefonts -qq\
    ffmpeg
    
RUN pip3 install --upgrade pip \
    && pip3 install jupyterlab 

    # refresh matplotlib cache
    # && rm -r ~/.cache/matplotlib


# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
