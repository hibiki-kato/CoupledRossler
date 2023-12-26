#include "myFunc.hpp"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>

namespace myfunc{
    void duration(std::chrono::time_point<std::chrono::system_clock> start){
        auto end = std::chrono::system_clock::now(); // 計測終了時間
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

    // ルンゲ＝クッタ法を用いた"ヤコビ行列"による時間発展
    Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt){
        Eigen::VectorXd k1, k2, k3, k4;
        Eigen::VectorXd nextState;
        
        k1 = dt * computeDerivativeJacobian(state, jacobian);
        k2 = dt * computeDerivativeJacobian(state + 0.5 * k1, jacobian);
        k3 = dt * computeDerivativeJacobian(state + 0.5 * k2, jacobian);
        k4 = dt * computeDerivativeJacobian(state + k3, jacobian);

        nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
        return nextState;
    }

    // ヤコビ行列による時間発展の一次近似
    Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian) {
        Eigen::VectorXd derivative(state.rows());
        derivative = jacobian * state;
        return derivative;
    }
    //　極大値を抽出するローレンツマップ
    Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj, int loc_max_dim){
        // 条件に合えば1, 合わなければ0のベクトルを作成
        std::vector<int> binLoc_max(traj.cols());
        //　最初の3点と最後の3点は条件を満たせないので0
        for (int i = 0; i < 3; ++i){
            binLoc_max[i] = 0;
            binLoc_max[binLoc_max.size()-1-i] = 0;
        }
        for (int i = 0; i < traj.cols()-6; ++i){
            //極大値か判定
            if (traj(loc_max_dim, i+1) - traj(loc_max_dim, i) > 0
            && traj(loc_max_dim, i+2) - traj(loc_max_dim, i+1) > 0
            && traj(loc_max_dim, i+3) - traj(loc_max_dim, i+2) > 0
            && traj(loc_max_dim, i+4) - traj(loc_max_dim, i+3) < 0
            && traj(loc_max_dim, i+5) - traj(loc_max_dim, i+4) < 0
            && traj(loc_max_dim, i+6) - traj(loc_max_dim, i+5) < 0){
                binLoc_max[i+3] = 1;
            } else{
                binLoc_max[i+3] = 0;
            }
        }
        //binLoc_maxの合計からloc_max_pointの列数を決定
        int count = std::accumulate(binLoc_max.begin(), binLoc_max.end(), 0);
        Eigen::MatrixXd loc_max_point(traj.rows(),count);
        int col_now = 0;
        for (int i = 0; i < binLoc_max.size(); ++i){
            if (binLoc_max[i] == 1){
                loc_max_point.col(col_now) = traj.col(i);
                col_now++;
            }
        }
        return loc_max_point;
}
}