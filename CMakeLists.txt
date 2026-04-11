cmake_minimum_required(VERSION 3.16)
project(lect2)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

set(FETCHCONTENT_QUIET FALSE)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_PROGRESS TRUE
    FETCHCONTENT_QUIET FALSE
)
FetchContent_MakeAvailable(nlohmann_json)


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(main main.cpp
    lib.cpp
)
target_link_libraries(main ${OpenCV_LIBS})
