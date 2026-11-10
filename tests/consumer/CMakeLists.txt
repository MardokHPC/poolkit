cmake_minimum_required(VERSION 3.18)

# Standalone project on purpose: it must not see poolkit's build tree, only the
# installed package. Configure it with -DCMAKE_PREFIX_PATH=<install prefix>.
project(poolkit_consumer LANGUAGES CXX)

# find_package with an explicit minimum version is what a consumer does, and the
# installed package is checked with SameMajorVersion: asking for 1.0 accepts
# 1.x.y and refuses 2.0.0, which is the contract the release note promises.
find_package(poolkit 1.0 CONFIG REQUIRED)

add_executable(poolkit_consumer main.cpp)
target_link_libraries(poolkit_consumer PRIVATE poolkit::poolkit)
