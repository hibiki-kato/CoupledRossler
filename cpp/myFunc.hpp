#include <iostream>
#include <chrono>
#include <cmath>

namespace myfunc{
    void duration(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end);
    int shift(double pre_theta, double theta, int rotation_number);
}
