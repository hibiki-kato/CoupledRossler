/**
 * @file lyapunov.cpp
 * @author Hibiki Kato
 * @brief compute lyapunov exponents using QR decomposition
 * @version 0.1
 * @date 2023-12-19
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>
#include <chrono>
#include <numeric>
#include <omp.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "Runge_Kutta.hpp"
#include "cnpy/cnpy.h"
#include "matplotlibcpp.h"
#include "Eigen_numpy_converter.hpp"
#include "myFunc.hpp"
namespace plt = matplotlibcpp;

using namespace Eigen;

// 関数プロトタイプ
VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian);
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt);


// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+5;
    double dump = 1e+3;
    double omega1 = 0.95;
    double omega2 = 0.99;
    double epsilon = 0.0416;
    double a = 0.165;
    double c = 10;
    double f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    
    int numthreads = omp_get_max_threads();
    CoupledRossler CR(omega1, omega2, epsilon, a, c, f, dt, t_0, t, dump, x_0);

    // std::string suffix = "laminar"; //ファイル名の後ろにつける文字列
    std::string suffix = ""; //ファイル名の後ろにつける文字列
    // 計算
    std::cout << "calculating trajectory" << std::endl;
    Eigen::MatrixXd traj =  CR.get_trajectory();
    // データの読み込みをここに記述
    // Eigen::MatrixXd traj = npy2EigenMat<double>("../../generated_lam/generated_laminar_beta_0.417nu_0.00018_dt0.01_50000period1300check200progresseps0.05.npy");
    
    Eigen::MatrixXd Data = traj.topRows(traj.rows() - 1);
    
    int dim = Data.rows() - 1;
    int numTimeSteps = Data.cols();
    int numVariables = Data.rows();

    //任意の直行行列を用意する
    MatrixXd Base = Eigen::MatrixXd::Random(numVariables, numVariables);
    HouseholderQR<MatrixXd> qr_1(Base);
    Base = qr_1.householderQ();
    // 総和の初期化
    VectorXd sum = Eigen::VectorXd::Zero(numVariables);
    // 次のステップ(QR分解されるもの)
    MatrixXd next(numVariables, numVariables);

    for (int i = 0; i < numTimeSteps; ++i) {
        std::cout << "\r processing..." << i << "/" << numTimeSteps << std::flush;
        // ヤコビアンの計算
        Eigen::MatrixXd jacobian = CR.jacobi_matrix(Data.col(i));
        // ヤコビアンとBase(直行行列)の積を計算する
        #pragma omp paralell for num_threads(threads)
        for (int j = 0; j < numVariables; ++j) {
            next.col(j) = rungeKuttaJacobian(Base.col(j), jacobian, dt); //線型ソルバで良い(要計算　計算コスト)
        }

        // QR分解を行う
        HouseholderQR<MatrixXd> qr(next);
        // 直交行列QでBaseを更新
        Base = qr.householderQ();
        // Rの対角成分を総和に加える
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
        // Rの対角成分の絶対値のlogをsumにたす
        Eigen::VectorXd diag = R.diagonal().cwiseAbs().array().log();
        sum += diag;

        // 進捗表示
        // if (i % 10000 == 0){
        //     std::cout << "\r" <<  sum(0) / (i+1) / dt << std::flush;
        // }
    }

    VectorXd lyapunovExponents = sum.array() / numTimeSteps / dt;

    // 結果の表示
    std::cout << lyapunovExponents.rows() << std::endl;
    std::cout << "Lyapunov Exponents:" << std::endl;
    std::cout << lyapunovExponents << std::endl;

    // plot settings
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "18";
    plt::rcparams(plotSettings);
    plt::figure_size(1200, 780);
    // 2からlyapunovExponents.rows()まで等差２の数列

    std::vector<int> xticks(lyapunovExponents.rows());
    std::iota(begin(xticks), end(xticks), 1);
    plt::xticks(xticks);
    // Add graph title
    std::vector<double> x(lyapunovExponents.data(), lyapunovExponents.data() + lyapunovExponents.size());

    plt::plot(xticks, x, "*-");
    // plt::ylim(-1, 1);
    plt::axhline(0, 0, lyapunovExponents.rows(), {{"color", "black"}, {"linestyle", "--"}});
    plt::xlabel("Number");
    plt::ylabel("Lyapunov Exponents");
    std::ostringstream oss;
    oss << "../../lyapunov/epsilon" << epsilon << "_a" << a << "_c" << c << "_f" << f << "_dt" << dt << "_t" << t << "_omega(" << omega1 << "," << omega2 << ")_" << suffix << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    // xをテキストファイルに保存
    oss.str("");
    oss << "../../lyapunov/epsilon" << epsilon << "_a" << a << "_c" << c << "_f" << f << "_dt" << dt << "_t" << t << "_omega(" << omega1 << "," << omega2 << ")_" << suffix << ".txt";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < lyapunovExponents.rows(); ++i) {
        ofs << lyapunovExponents(i) << std::endl;
    }
    ofs.close();

    myfunc::duration(start, std::chrono::system_clock::now()); // 計測終了時間
}

// ルンゲ＝クッタ法を用いた"ヤコビアン"による時間発展
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt){
    VectorXd k1, k2, k3, k4;
    VectorXd nextState;
    
    k1 = dt * computeDerivativeJacobian(state, jacobian);
    k2 = dt * computeDerivativeJacobian(state + 0.5 * k1, jacobian);
    k3 = dt * computeDerivativeJacobian(state + 0.5 * k2, jacobian);
    k4 = dt * computeDerivativeJacobian(state + k3, jacobian);

    nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

    return nextState;
}

VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian) {
    VectorXd derivative(state.rows());
    derivative = jacobian * state;
    return derivative;
}