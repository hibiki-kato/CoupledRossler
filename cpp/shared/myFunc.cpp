#include "myFunc.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

namespace myfunc{
    std::string ordinal_suffix(int n) {
    return std::to_string(n) + (n % 100 < 11 || n % 100 > 13 ? (n % 10 == 1 ? "st" : (n % 10 == 2 ? "nd" : (n % 10 == 3 ? "rd" : "th"))) : "th");
    }
    void duration(std::chrono::time_point<std::chrono::system_clock> start){
        auto end = std::chrono::system_clock::now(); // 計測終了時間
        int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
        int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
        int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
        std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    }
}