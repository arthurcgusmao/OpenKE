release_dir="./release"

if [ ! -d "${release_dir}" ]; then
    mkdir $release_dir
fi

g++ ./base/Base.cpp -std=c++03 -fPIC -shared -o ./release/Base.so -pthread -O3 -march=native
