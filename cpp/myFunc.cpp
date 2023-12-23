#include "myFunc.hpp"
#include <iostream>
#include <chrono>
#include <cmath>

namespace myfunc{
    void duration(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end){
        int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
        int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
        int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
        int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
        std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    }

    /**
     * @brief given previous theta and rotation_number and current theta,  return rotation number(unwrapped)
     * 
     * @param pre_theta : previous theta
     * @param theta : current theta
     * @param rotation_number : previous rotation number (n in Z, unwrapped angle is theta + 2 * n * pi)
     * @return int 
     */
    int shift(double pre_theta, double theta, int rotation_number){
        //forward
        if ((theta - pre_theta) < -M_PI){
            rotation_number += 1;
        }
        //backward
        else if ((theta - pre_theta) > M_PI){
            rotation_number -= 1;
        }

        return rotation_number;
    }
    
}