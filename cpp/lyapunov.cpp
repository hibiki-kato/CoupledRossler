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
#include <random>
#include <chrono>
#include <numeric>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include "shared/Flow.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
#include "shared/myFunc.hpp"
namespace plt = matplotlibcpp;

// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double dt = 0.01;
    double t_0 = 0;
    double t = 2e+5;
    double dump = 1e+4;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.038;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/chaotic.npy", true);
    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<> dist(-10, 10);
    for (int i = 0; i < x_0.rows(); ++i) {
        x_0(i) = dist(engine);
    }
    
    int numThreads = 1; //正確な計算を行うためには1スレッドで実行する必要がある
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    // 計算
    // std::cout << "calculating trajectory" << std::endl;
    // Eigen::MatrixXd traj =  CR.get_trajectory();
    // std::string suffix = ""; //ファイル名の後ろにつける文字列
    // データの読み込みをここに記述
    Eigen::MatrixXd traj = npy2EigenMat<double>("../generated_lam/sync_gen_laminar_epsilon0.038_a0.165_c10_f0.2_omega0.95-0.99_t2000001500check200progress10^-16-10^-9perturb.npy", true);
    std::string suffix = "laminar"; //ファイル名の後ろにつける文字列

    Eigen::MatrixXd Data = traj.topRows(traj.rows() - 1);
    
    int dim = Data.rows() - 1;
    int numTimeSteps = Data.cols();
    int numVariables = Data.rows();

    // 軌道の確認
    std::vector<double> x1(numTimeSteps), x2(numTimeSteps);
    for (int i = 0; i < numTimeSteps; ++i) {
        x1[i] = Data(0, i);
        x2[i] = Data(3, i);
    }
    plt::scatter(x1, x2);
    plt::save("traj.png");
    plt::clf();
    std::cout << "check traj.png" << std::endl;

    std::cout << "calculating lyapunov exponents" << std::endl;
    //DataをnumThreads個に分割する(実際に分割しないが，分割したときのインデックスを計算する)
    std::vector<int> splitIndex(numThreads + 1);
    splitIndex[0] = 0;
    splitIndex[numThreads] = numTimeSteps;
    for (int i = 1; i < numThreads; ++i) {
        splitIndex[i] = numTimeSteps / numThreads * i;
    }
    

    //任意の直行行列を用意する
    Eigen::MatrixXd Base = Eigen::MatrixXd::Random(numVariables, numVariables);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr_1(Base);
    Base = qr_1.householderQ();
    // 総和の初期化
    Eigen::VectorXd sum = Eigen::VectorXd::Zero(numVariables);
    // 次のステップ(QR分解されるもの)
    Eigen::MatrixXd next(numVariables, numVariables);
    
    #pragma omp declare reduction(vec_add : Eigen::VectorXd : omp_out += omp_in) \
        initializer(omp_priv = Eigen::VectorXd::Zero(omp_orig.size()))
    
    #pragma omp parallel for num_threads(numThreads) firstprivate(CR, next, Base, dt) shared(Data, splitIndex) reduction(vec_add:sum)
    for (int i = 0; i < numThreads; ++i) {
        for (int j = splitIndex[i]; j < splitIndex[i + 1]; ++j) {
            // 進捗の表示
            if (i == numThreads - 1){
                if (j % (numTimeSteps/10000) == 0){
                    std::cout << "\r" <<  (j - splitIndex[numThreads - 1]) / static_cast<double>(splitIndex[numThreads] - splitIndex[numThreads - 1]) * 100 << "%" << std::flush;
                }
            }
            // ヤコビアンの計算
            Eigen::MatrixXd jacobian = CR.jacobi_matrix(Data.col(j));
            // ヤコビアンとBase(直行行列)の積を計算する
            for (int k = 0; k < numVariables; ++k) {
                next.col(k) = myfunc::rungeKuttaJacobian(Base.col(k), jacobian, dt);
            }
            // QR分解を行う
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(next);
            // 直交行列QでBaseを更新
            Base = qr.householderQ();
            // Rの対角成分を総和に加える
            Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
            // Rの対角成分の絶対値のlogをsumにたす
            Eigen::VectorXd diag = R.diagonal().cwiseAbs().array().log();

            sum += diag;
            // 途中経過の表示
            // if (j % 10000 == 0){
            //     std::cout << "\r" <<  sum(0) / (j+1) / dt << std::endl;
            // }
        }
    }

    Eigen::VectorXd lyapunovExponents = sum.array() / (numTimeSteps * dt); // 1秒あたりの変化量に変換

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
    oss << "../../lyapunov/img/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << suffix << ".png";  // 文字列を結合する
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    // xをテキストファイルに保存
    oss.str("");
    oss << "../../lyapunov/epsilon" << params.epsilon << "_t" << t << "_a" << params.a << "_c" << params.c << "_f" << params.f << "_omega" << params.omega1 << "-" << params.omega2 << "_dt" << dt << "_dump" << dump << suffix << ".txt";  // 文字列を結合する
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < lyapunovExponents.rows(); ++i) {
        ofs << lyapunovExponents(i) << std::endl;
    }
    ofs.close();

    myfunc::duration(start); // 計測終了時間
}