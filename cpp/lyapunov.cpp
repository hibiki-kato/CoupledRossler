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
namespace plt = matplotlibcpp;

using namespace Eigen;

// 関数プロトタイプ
MatrixXd computeJacobian(const VectorXd& state, Eigen::VectorXd k_n, double beta, double nu);
VectorXd computeDerivativeJacobian(const VectorXd& state, const MatrixXd& jacobian);
VectorXd rungeKuttaJacobian(const VectorXd& state, const MatrixXd& jacobian, double dt);
// VectorXd rungeKuttaStep(const VectorXd& state, double dt, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f);
// Eigen::VectorXcd goy_shell_model(Eigen::VectorXd state, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f);


// メイン関数
int main() {
    auto start = std::chrono::system_clock::now(); // 計測開始時間
    double nu = 0.00018;
    double beta = 0.417;
    std::complex<double> f = std::complex<double>(1.0,1.0) * 5.0 * 0.001;
    double dt = 0.01;
    double t_0 = 0;
    double t = 5e+4;
    double latter = 1;
    int threads = omp_get_max_threads();
    Eigen::VectorXcd x_0 = npy2EigenVec<std::complex<double>>("../../initials/beta0.44_nu0.00018_10000period_dt0.01_4-7_4-10_4-13_7-10_7-13_10-13_5-8_5-11_5-14_8-11_8-14_11-14_6-9_6-12_9-12.npy");
    
    ShellModel SM(nu, beta, f, dt, t_0, t, latter, x_0);
    std::cout << "calculating trajectory" << std::endl;
    // bool laminar = false;
    // Eigen::MatrixXcd rawData = SM.get_trajectory_();
    // データの読み込みをここに記述
    bool laminar = true;
    Eigen::MatrixXcd rawData = npy2EigenMat<std::complex<double>>("../../generated_lam/generated_laminar_beta_0.417nu_0.00018_dt0.01_50000period1300check200progresseps0.05.npy");
    
    
    int dim = rawData.rows() - 1;
    // データの整形(実関数化)
    Eigen::MatrixXd Data(dim*2, rawData.cols());
    for (int i = 0; i < dim; ++i) {
        Data.row(2*i) = rawData.row(i).real();
        Data.row(2*i+1) = rawData.row(i).imag();
    }

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
        VectorXd state = Data.col(i);
        // ヤコビアンの計算
        Eigen::MatrixXd jacobian = computeJacobian(state, SM.get_k_n_(), SM.get_beta_(), SM.get_nu_());
        // ヤコビアンとBase(直行行列)の積を計算する
        #pragma omp paralell for num_threads(threads)
        for (int j = 0; j < numVariables; ++j) {
            next.col(j) = rungeKuttaJacobian(Base.col(j), jacobian, dt);
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
        if (i % 10000 == 0){
            std::cout << "\r" <<  sum(0) / (i+1) / dt << std::flush;
        }

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
    plt::xlabel("wavenumber");
    plt::ylabel("Lyapunov Exponents");
    std::ostringstream oss;
    if (laminar) {
        oss << "../../lyapunov_exponents/beta_" << beta << "nu_" << nu <<"_"<< static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period_laminar.png";  // 文字列を結合する
    } else {
        oss << "../../lyapunov_exponents/beta_" << beta << "nu_" << nu <<"_"<< static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period.png";  // 文字列を結合する
    }
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    // xをテキストファイルに保存
    oss.str("");
    if (laminar) {
        oss << "../../lyapunov_exponents/beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period_laminar.txt";
    } else {
        oss << "../../lyapunov_exponents/beta_" << beta << "nu_" << nu <<"_dt"<< dt << "_" << static_cast<int>(rawData.cwiseAbs().bottomRightCorner(1, 1)(0, 0)) << "period.txt";
    }
    std::string fname = oss.str(); // 文字列を取得する
    std::cout << "saving as " << fname << std::endl;
    std::ofstream ofs(fname);
    for (int i = 0; i < lyapunovExponents.rows(); ++i) {
        ofs << lyapunovExponents(i) << std::endl;
    }
    ofs.close();

    auto end = std::chrono::system_clock::now();  // 計測終了時間
    int hours = std::chrono::duration_cast<std::chrono::hours>(end-start).count(); //処理に要した時間を変換
    int minutes = std::chrono::duration_cast<std::chrono::minutes>(end-start).count(); //処理に要した時間を変換
    int seconds = std::chrono::duration_cast<std::chrono::seconds>(end-start).count(); //処理に要した時間を変換
    int milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); //処理に要した時間を変換
    std::cout << hours << "h " << minutes % 60 << "m " << seconds % 60 << "s " << milliseconds % 1000 << "ms " << std::endl;
    return 0;
}

// ヤコビアンの計算
MatrixXd computeJacobian(const VectorXd& state, Eigen::VectorXd k_n, double beta, double nu){
    int dim = state.rows();
    MatrixXd jacobian = Eigen::MatrixXd::Zero(dim, dim);
    
    // A
    for (int i = 0; i < dim/2 - 2; ++i) {
        jacobian(2*i, 2*i + 2) += k_n(i) * state((i+2)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i + 2) += k_n(i) * state((i+2)*2);
        jacobian(2*i+1, 2*i+1 + 2) += -k_n(i) * state((i+2)*2 + 1);

        jacobian(2*i, 2*i + 4) +=  k_n(i) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 + 4) += k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i + 4) +=  k_n(i) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 + 4) += -k_n(i) * state((i+1)*2 + 1);
    }

    // B
    for (int i = 1; i < dim/2 - 1; ++i) {
        jacobian(2*i, 2*i - 2) +=  -beta * k_n(i-1) * state((i+1)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i - 2) += -beta * k_n(i-1) * state((i+1)*2);
        jacobian(2*i+1, 2*i+1 - 2) += beta * k_n(i-1) * state((i+1)*2 + 1);

        jacobian(2*i, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 + 2) += -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i + 2) +=  -beta * k_n(i-1) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 + 2) += beta * k_n(i-1) * state((i-1)*2 + 1);
    }

    // Gamma
    for (int i = 2; i < dim/2; ++i) {
        jacobian(2*i, 2*i - 4) +=  (beta-1) * k_n(i-2) * state((i-1)*2 + 1);
        jacobian(2*i, 2*i+1 - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i - 4) += (beta-1) * k_n(i-2) * state((i-1)*2);
        jacobian(2*i+1, 2*i+1 - 4) += (1-beta) * k_n(i-2) * state((i-1)*2 + 1);

        jacobian(2*i, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2 + 1);
        jacobian(2*i, 2*i+1 - 2) += (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i - 2) +=  (beta-1) * k_n(i-2) * state((i-2)*2);
        jacobian(2*i+1, 2*i+1 - 2) += (1-beta) * k_n(i-2) * state((i-2)*2 + 1);
    }
    
    // N
    for (int i = 0; i < dim/2; ++i) {
        jacobian(2*i, 2*i) = -nu*k_n(i)*k_n(i);
        jacobian(2*i+1, 2*i+1) = -nu*k_n(i)*k_n(i);
    }

    return jacobian;
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

// // ルンゲクッタ法を用いた時間発展
// VectorXd rungeKuttaStep(const VectorXd& state, double dt, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f){
//     VectorXd k1, k2, k3, k4;
//     VectorXd nextState;

//     k1 = dt * goy_shell_model(state, c_n_1, c_n_2, c_n_3, k_n, nu, f);
//     k2 = dt * goy_shell_model(state + 0.5 * k1, c_n_1, c_n_2, c_n_3, k_n, nu, f);
//     k3 = dt * goy_shell_model(state + 0.5 * k2, c_n_1, c_n_2, c_n_3, k_n, nu, f);
//     k4 = dt * goy_shell_model(state + k3, c_n_1, c_n_2, c_n_3, k_n, nu, f);

//     nextState = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

//     return nextState;
// }

// Eigen::VectorXcd goy_shell_model(Eigen::VectorXd state, Eigen::VectorXd c_n_1, Eigen::VectorXd c_n_2, Eigen::VectorXd c_n_3, Eigen::VectorXd k_n, double nu, std::complex<double> f){
//     //convert real to complex
//     int dim = state.rows()/2;
//     Eigen::VectorXcd u = Eigen::VectorXcd::Zero(dim+4);
//     Eigen::VectorXd state_real(dim);
//     Eigen::VectorXd state_imag(dim);
//     for(int i = 0; i < dim; i++){
//         state_real(i) = state(2*i);
//         state_imag(i) = state(2*i+1);
//     }
//     u.middleRows(2, dim).real() = state_real;
//     u.middleRows(2, dim).imag() = state_imag;

//     // compute complex
//     Eigen::VectorXcd ddt_u_complex = (c_n_1.array() * u.middleRows(3,dim).conjugate().array() * u.bottomRows(dim).conjugate().array()
//                             + c_n_2.array() * u.middleRows(1,dim).conjugate().array() * u.middleRows(3,dim).conjugate().array()
//                             + c_n_3.array() * u.middleRows(1,dim).conjugate().array() * u.topRows(dim).conjugate().array()) * std::complex<double>(0, 1.0)
//                             - nu * u.middleRows(2,dim).array() * k_n.array().square();
//     ddt_u_complex(0) += f;

//     // convert complex to real
//     Eigen::VectorXd ddt_u_real(dim*2);
//     for(int i = 0; i < dim; i++){
//         ddt_u_real(2*dim) = ddt_u_complex(i).real();
//         ddt_u_real(2*i+1) = ddt_u_complex(i).imag();
//     }


//     return ddt_u_real;
// }