#include <iostream>
#include <fstream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <random>
#include "shared/Flow.hpp"
#include <chrono>
#include "shared/myFunc.hpp"
#include "shared/Flow.hpp"
#include "shared/Map.hpp"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/matplotlibcpp.h"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // timer start