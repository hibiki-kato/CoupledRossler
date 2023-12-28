/**
 * @file Flow.cpp
 * @author Hibiki Kato
 * @brief flow classes
 * @version 0.1
 * @date 2023-12-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#include "Flow.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <random>

/*
  █████    ██                  ██   ██   ███        ███                     █             ██
 ████████  ██                  ██   ██   ████       ███                     █             ██
██     ██  ██                  ██   ██   ████       ███                     █             ██
██      ██ ██                  ██   ██   ████      ████                     █             ██
██         ██ ████     ████    ██   ██   ██ ██     ████     █████      ██████     ████    ██
 ██        ████████   ███████  ██   ██   ██ ██     █ ██    ███████    ███████    ███████  ██
  ████     ██    ██  ██    ██  ██   ██   ██  █    ██ ██   ██    ███  ██    ██   ██    ██  ██
    ████   ██    ██  ██     █  ██   ██   ██  ██   ██ ██   ██     ██  ██     █   ██     █  ██
      ███  ██    ██  ████████  ██   ██   ██  ██  ██  ██   ██     ██  ██     █   ████████  ██
       ███ ██    ██  ██        ██   ██   ██   ██ ██  ██   ██     ██  ██     █   ██        ██
██      ██ ██    ██  ██        ██   ██   ██   ██ █   ██   ██     ██  ██     █   ██        ██
██     ██  ██    ██  ██        ██   ██   ██    ███   ██   ██    ███  ██    ██   ██        ██
 ████████  ██    ██   ███████  ██   ██   ██    ███   ██    ███████    ███████    ███████  ██
  █████    ██    ██    █████   ██   ██   ██    ██    ██     █████      ████ █     █████   ██
*/
ShellModel::ShellModel(SMparams input_params, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXcd input_x_0){
    dt = input_dt;
    t_0 = input_t_0;
    t = input_t;
    dump = input_dump;
    x_0 = input_x_0;
    //parameters
    nu = input_params.nu;
    beta = input_params.beta;
    f = input_params.f;
    
    int dim = x_0.rows();

    // make k_n and c_n using beta
    k_n = Eigen::VectorXd::Zero(dim);
    double q = 2.0;
    double k_0 = pow(2, -4);
    
    for (int i = 0; i < dim; i++) {
        k_n(i) = k_0 * pow(q, i+1);
    };
    c_n_1 = Eigen::VectorXd::Zero(dim);
    c_n_1.topRows(dim-2) = k_n.topRows(dim-2);

    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.topRows(dim-2).array() * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.topRows(dim-2).array() * (beta - 1);
    steps = static_cast<long>((t - t_0) / dt + 0.5);
    dump_steps = static_cast<long>(dump / dt + 0.5);
 }
//destructor
ShellModel::~ShellModel(){
 }

Eigen::MatrixXcd ShellModel::get_trajectory(){
    int row = x_0.rows() + 1;
    Eigen::MatrixXcd trajectory(row, steps+1);

    //set initial point
    trajectory.block(0, 0, row-1, 1) = x_0;
    //renew x_0 (dump)
    for (long i = 0; i < dump_steps; i++){
        trajectory.topLeftCorner(row-1, 1) = ShellModel::rk4(trajectory.topLeftCorner(row-1, 1));
    }

    double time = t_0;
    trajectory(row-1, 0) = time;
    //solve
    for(long i = 0; i < steps; i++){
        time += dt;
        trajectory.block(0, i+1, row-1, 1) = ShellModel::rk4(trajectory.block(0, i, row-1, 1));
        trajectory(row-1, i+1) = time;
    }
    return trajectory;
};

Eigen::VectorXd ShellModel::energy_spectrum(const Eigen::MatrixXcd& trajectory){
    if (trajectory.rows() == 0){
        Eigen::VectorXcd x = x_0;
        Eigen::VectorXd sum(x_0.rows(), 1);
        
        //renew x_0 while reaching latter
        for (long long i = 0; i < dump_steps; i++){
            x = ShellModel::rk4(x);
        }

        // get energy spectrum by calc mean of absolute value of each shell's
        sum = x.cwiseAbs();
        for (long long i = 0; i < steps; i++){
            x = ShellModel::rk4(x);
            sum += x.cwiseAbs();
        }
        return sum.array() / (steps+1);
    }
    else{
        Eigen::VectorXd sum(trajectory.rows()-1, 1);
        sum = trajectory.topRows(trajectory.rows()-1).cwiseAbs().rowwise().mean();
        return sum;
    }
    
}

void ShellModel::set_beta_(double input_beta){
    beta = input_beta;
    int dim = x_0.rows();
    // update c_n_2 and c_n_3
    c_n_2 = Eigen::VectorXd::Zero(dim);
    c_n_2.middleRows(1, dim-2) = k_n.topRows(dim-2).array() * (-beta);

    c_n_3 = Eigen::VectorXd::Zero(dim);
    c_n_3.bottomRows(dim-2) = k_n.topRows(dim-2).array() * (beta - 1);
}
void ShellModel::set_t_0_(double input_t_0){
    t_0 = input_t_0;
    steps = static_cast<long>((t - t_0) / dt);
}
void ShellModel::set_t_(double input_t){
    t = input_t;
    steps = static_cast<long>((t - t_0) / dt);
}


Eigen::VectorXcd ShellModel::rk4(const Eigen::VectorXcd& present){
    Eigen::VectorXcd k1 = dt * ShellModel::goy_shell_model(present).array();
    Eigen::VectorXcd k2 = dt * ShellModel::goy_shell_model(present.array() + k1.array() /2).array();
    Eigen::VectorXcd k3 = dt * ShellModel::goy_shell_model(present.array() + k2.array() /2).array();
    Eigen::VectorXcd k4 = dt * ShellModel::goy_shell_model(present.array() + k3.array()).array();
    return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
}

Eigen::VectorXcd ShellModel::goy_shell_model(const Eigen::VectorXcd& state){
    int dim = state.rows();
    Eigen::VectorXcd u = Eigen::VectorXd::Zero(dim+4);

    u.middleRows(2, state.rows()) = state;
    Eigen::VectorXcd dt_u = (c_n_1.array() * u.middleRows(3,dim).conjugate().array() * u.bottomRows(dim).conjugate().array()
                            + c_n_2.array() * u.middleRows(1,dim).conjugate().array() * u.middleRows(3,dim).conjugate().array()
                            + c_n_3.array() * u.middleRows(1,dim).conjugate().array() * u.topRows(dim).conjugate().array()) * std::complex<double>(0, 1.0)
                            - nu * u.middleRows(2,dim).array() * k_n.array().square();
    dt_u(0) += f;
    return dt_u;
}

Eigen::MatrixXd ShellModel::jacobian_matrix(const Eigen::VectorXd& state){
    int dim = state.rows();
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(dim, dim);
    
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

/*
   █████                                    ██                    █   ████████                                   ██
  ███████                                   ██                    █   █████████                                  ██
 ██     ██                                  ██                    █   ██     ███                                 ██
 ██     ██                                  ██                    █   ██      ██                                 ██
██       █    █████    ██    ██  ██ ████    ██     ████      ██████   ██      ██    █████      ████      ████    ██     ████    ██ ██
██           ███████   ██    ██  ████████   ██    ███████   ███████   ██      ██   ███████    ██████    ██████   ██    ███████  █████
██          ██    ███  ██    ██  ██    ██   ██   ██    ██  ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██    ██  ██
██          ██     ██  ██    ██  ██     ██  ██   ██     █  ██     █   ████████    ██     ██  ██        ██        ██   ██     █  ██
██          ██     ██  ██    ██  ██     ██  ██   ████████  ██     █   ███████     ██     ██   ████      ████     ██   ████████  ██
██       █  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██    ██    ██     ██     ████      ████   ██   ██        ██
 ██     ██  ██     ██  ██    ██  ██     ██  ██   ██        ██     █   ██     ██   ██     ██        ██        ██  ██   ██        ██
 ██     ██  ██    ███  ██    ██  ██    ██   ██   ██        ██    ██   ██     ██   ██    ███  ██    ██  ██    ██  ██   ██        ██
  ███████    ███████    ███████  ████████   ██    ███████   ███████   ██      ██   ███████   ███████   ███████   ██    ███████  ██
   █████      █████      ██████  ██ ████    ██     █████     ████ █   ██      ██    █████      ████      ████    ██     █████   ██
                                 ██
                                 ██
                                 ██
                                 ██
*/
//constructor
CoupledRossler::CoupledRossler(CRparams input_params, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0){
    dt = input_dt;
    t_0 = input_t_0;
    t = input_t;
    dump = input_dump;
    x_0 = input_x_0;

    //parameters
    omega1 = input_params.omega1;
    omega2 = input_params.omega2;
    epsilon = input_params.epsilon;
    a = input_params.a;
    c = input_params.c;
    f = input_params.f;
    
    int dim = 6;
    
    steps = static_cast<long long>((t - t_0) / dt+ 0.5);
    dump_steps = static_cast<long long>(dump / dt + 0.5);
 }
//destructor
CoupledRossler::~CoupledRossler(){
}

Eigen::MatrixXd CoupledRossler::get_trajectory(){
    int row = 6 + 1;
    Eigen::MatrixXd trajectory(row, steps+1);

    //set initial point
    trajectory.topLeftCorner(row - 1, 1) = x_0;
    //renew x_0 (dump)
    for (long long i = 0; i < dump_steps; i++){
        trajectory.topLeftCorner(row - 1, 1) = CoupledRossler::rk4(trajectory.topLeftCorner(row - 1, 1));
    }
    //solve
    double time = t_0;
    trajectory(row-1, 0) = time;
    for(long long i = 0; i < steps; i++){
        time += dt;
        trajectory.block(0, i+1, row-1, 1) = CoupledRossler::rk4(trajectory.block(0, i, row-1, 1));
        trajectory(row-1, i+1) = time;
    }
    return trajectory;
}

Eigen::VectorXd CoupledRossler::rk4(const Eigen::VectorXd& present){
    Eigen::VectorXd k1 = dt * CoupledRossler::coupled_rossler(present);
    Eigen::VectorXd k2 = dt * CoupledRossler::coupled_rossler(present.array() + k1.array() /2);
    Eigen::VectorXd k3 = dt * CoupledRossler::coupled_rossler(present.array() + k2.array() /2);
    Eigen::VectorXd k4 = dt * CoupledRossler::coupled_rossler(present.array() + k3.array());
    return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
}

Eigen::VectorXd CoupledRossler::coupled_rossler(const Eigen::VectorXd& state){
    double x1 = state(0);
    double y1 = state(1);
    double z1 = state(2);
    double x2 = state(3);
    double y2 = state(4);
    double z2 = state(5);

    double dx1 = -omega1 * y1 - z1 + epsilon * (x2 - x1);
    double dy1 = omega1 * x1 + a * y1;
    double dz1 = f + z1 * (x1 - c);

    double dx2 = -omega2 * y2 - z2 + epsilon * (x1 - x2);
    double dy2 = omega2 * x2 + a * y2;
    double dz2 = f + z2 * (x2 - c);

    //dx1からdz2をベクトルにまとめる
    Eigen::VectorXd dt_f(6);
    dt_f << dx1, dy1, dz1, dx2, dy2, dz2;
    return dt_f;
}

Eigen::MatrixXd CoupledRossler::jacobi_matrix(const Eigen::VectorXd& state){
    double x1 = state(0);
    double y1 = state(1);
    double z1 = state(2);
    double x2 = state(3);
    double y2 = state(4);
    double z2 = state(5);

    Eigen::MatrixXd J(6, 6);
    J << -epsilon, -omega1, -1, epsilon, 0, 0,
        omega1, a, 0, 0, 0, 0,
        z1, 0, x1-c, 0, 0, 0,
        epsilon, 0, 0, -epsilon, -omega2, -1,
        0, 0, 0, omega2, a, 0,
        0, 0, 0, z2, 0, x2-c;
    return J;
}


/*
       ██                                 ██         ██                            ███        ███                         ██
       ██                                 ██         ██                            ████       ███                         ██
       ██                                 ██                                       ████       ███              ██
       ██                                 ██                                       ████      ████              ██
       ██    ████      ████      █████    ██ ████    ██     ████    ██ ████        ██ ██     ████     ████   ██████ ██ ██ ██   ██    ██
       ██   ███ ██    ███████   ███████   ████████   ██    ███ ██   ████████       ██ ██     █ ██    ███ ██    ██   █████ ██   ██   ██
       ██  ██    ██  ██    ██  ██    ███  ██    ██   ██   ██    ██  ██    ██       ██  █    ██ ██   ██    ██   ██   ██    ██    ██ ██
       ██        ██  ██     █  ██     ██  ██     ██  ██         ██  ██    ██       ██  ██   ██ ██         ██   ██   ██    ██     ████
       ██    ██████  ██        ██     ██  ██     ██  ██     ██████  ██    ██       ██  ██  ██  ██     ██████   ██   ██    ██     ███
       ██  ███   ██  ██        ██     ██  ██     ██  ██   ███   ██  ██    ██       ██   ██ ██  ██   ███   ██   ██   ██    ██     ███
██     ██  ██    ██  ██        ██     ██  ██     ██  ██   ██    ██  ██    ██       ██   ██ █   ██   ██    ██   ██   ██    ██     ████
 ██    ██  ██    ██  ██    ██  ██    ███  ██    ██   ██   ██    ██  ██    ██       ██    ███   ██   ██    ██   ██   ██    ██    ██ ██
 ███████   ████████   ███████   ███████   ████████   ██   ████████  ██    ██       ██    ███   ██   ████████   ███  ██    ██   ██   ██
  █████     ████ ██    ████      █████    ██ ████    ██    ████ ██  ██    ██       ██    ██    ██    ████ ██    ███ ██    ██   ██    ██
*/

namespace myfunc{
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

    Eigen::MatrixXd regularizeJacobian(const Eigen::MatrixXd& jacobian){
        //　ヤコビ行列の最大固有値を計算
        Eigen::EigenSolver<Eigen::MatrixXd> es(jacobian);
        Eigen::VectorXd eigenvalues = es.eigenvalues().real(); // 固有値(実数部のみ)
        double max_eigenvalue = eigenvalues.maxCoeff(); // 最大固有値
        return jacobian / max_eigenvalue; //最大伸び率=1に正規化
    }
}
/*
████████    ██
█████████   ██
██      ██  ██
██      ██  ██
██      ██  ██ ████     ████      ████      ████
██      ██  ████████   ███ ██    ██████    ███████
██     ███  ██    ██  ██    ██  ██    ██  ██    ██
█████████   ██    ██        ██  ██        ██     █
██████      ██    ██    ██████   ████     ████████
██          ██    ██  ███   ██     ████   ██
██          ██    ██  ██    ██        ██  ██
██          ██    ██  ██    ██  ██    ██  ██
██          ██    ██  ████████  ███████    ███████
██          ██    ██   ████ ██    ████      █████
*/

namespace myfunc{
    double shift(double pre_theta, double theta, double rotation_number){
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

    bool isSync(double a, double b, double sync_criteria, double center) {
        double lowerBound = center - sync_criteria;
        double upperBound = center + sync_criteria;
        long long n = 0;
        double diff = std::abs(a - b);
        // std::cout << diff << std::endl;
        while (lowerBound <= diff) {
            if (lowerBound <= diff && diff <= upperBound) {
                return true;
            }
            n++;
            lowerBound += 2  * M_PI;
            upperBound += 2  * M_PI;
        }
        return false;
    }
}