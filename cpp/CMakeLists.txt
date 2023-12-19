# CMakeのバージョンを設定
cmake_minimum_required(VERSION 3.14)
project(CoupledRossler)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 23)
set(EXECUTABLE exe)
add_executable(${EXECUTABLE} 
    lyapunov.cpp
    CoupledRossler.cpp
    myFunc.cpp
)

# message(STATUS "CMAKE_VERSION:${CMAKE_VERSION}" )
# message(STATUS "include_directories(PYTHON_INCLUDE_DIRS):${PYTHON_INCLUDE_DIRS}" )
# message(STATUS "include_directories(Python_INCLUDE_DIRS):${Python_INCLUDE_DIRS}" )
# message(STATUS "include_directories(PYTHON_LIBRARIES):${PYTHON_LIBRARIES}" )
# message(STATUS "include_directories(Python_LIBRARIES):${Python_LIBRARIES}" )
# message(STATUS "include_directories(Python_LIBRARY_DIRS):${Python_LIBRARY_DIRS}" )

# OSでソースが異なるので分岐
if($ENV{OS_NAME} STREQUAL "Darwin")
    message("Build in Mac")
    find_package (Python 3.10 REQUIRED COMPONENTS Interpreter Development NumPy)
    link_directories(/opt/homebrew/lib)
    include_directories(/opt/homebrew/include ${Python_INCLUDE_DIRS})
else()
    message("Build in Linux")
    find_package (Python 3 REQUIRED COMPONENTS Interpreter Development NumPy)
    link_directories(/usr/lib)
    include_directories(/usr/include)
    target_link_libraries(${EXECUTABLE} PUBLIC z cnpy)
endif()

#ライブラリを作成
add_library(cnpy "cnpy/cnpy.cpp")
target_link_libraries(${EXECUTABLE} PUBLIC z cnpy Python::Python Python::NumPy)

#キャッシュなし
set(CMAKE_NO_CACHE TRUE)
set(CMAKE_CXX_FLAGS "-O3 -fopenmp -mtune=native -march=native")
