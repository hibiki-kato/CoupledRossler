#include "Runge_Kutta.hpp"
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <math.h>
#include <random>

//constructor
CoupledRossler::CoupledRossler(double input_omega1, double input_omega2, double input_epsilon, double input_a, double input_c, double input_f, double input_dt, double input_t_0, double input_t, double input_dump, Eigen::VectorXd input_x_0){
    dt = input_dt;
    t_0 = input_t_0;
    t = input_t;
    dump = input_dump;
    x_0 = input_x_0;

    //parameters
    omega1 = input_omega1;
    omega2 = input_omega2;
    epsilon = input_epsilon;
    a = input_a;
    c = input_c;
    f = input_f;
    
    int dim = 6;
    
    steps = static_cast<long>((t - t_0) / dt+ 0.5);
    dump_steps = static_cast<long>(dump / dt + 0.5);
 }
//destructor
CoupledRossler::~CoupledRossler(){
}

Eigen::MatrixXd CoupledRossler::get_trajectory_(){
    int row = 6 + 1;
    Eigen::MatrixXd trajectory(row, steps+1);

    //set initial point
    trajectory.topLeftCorner(row - 1, 1) = x_0;
    //renew x_0 while reaching latter (dump)
    for (long i = 0; i < dump_steps; i++){
        trajectory.topLeftCorner(row - 1, 1) = CoupledRossler::rk4_(trajectory.topLeftCorner(row - 1, 1));
    }
    //solve
    double time = t_0;
    trajectory(row-1, 0) = time;
    for(long i = 0; i < steps; i++){
        time += dt;
        trajectory.block(0, i+1, row-1, 1) = CoupledRossler::rk4_(trajectory.block(0, i, row-1, 1));
        trajectory(row-1, i+1) = time;
    }
    return trajectory;
}

Eigen::VectorXd CoupledRossler::rk4_(const Eigen::VectorXd& present){
    Eigen::VectorXd k1 = dt * CoupledRossler::coupledRossler(present);
    Eigen::VectorXd k2 = dt * CoupledRossler::coupledRossler(present.array() + k1.array() /2);
    Eigen::VectorXd k3 = dt * CoupledRossler::coupledRossler(present.array() + k2.array() /2);
    Eigen::VectorXd k4 = dt * CoupledRossler::coupledRossler(present.array() + k3.array());
    return present.array() + (k1.array() + 2 * k2.array() + 2 * k3.array() + k4.array()) / 6;
}

Eigen::VectorXd CoupledRossler::coupledRossler(const Eigen::VectorXd& state){
    double x1 = state(0);
    double y1 = state(1);
    double z1 = state(2);
    double x2 = state(3);
    double y2 = state(4);
    double z2 = state(5);

    double dx1 = -omega1 * y1 - z1 + epsilon * (x2 - x1);
    double dy1 = omega1 * x1 + 0.165 * y1;
    double dz1 = f + z1 * (x1 - 10);

    double dx2 = -omega2 * y2 - z2 + epsilon * (x1 - x2);
    double dy2 = omega2 * x2 + a * y2;
    double dz2 = f + z2 * (x2 - c);

    //dx1からdz2をベクトルにまとめる
    Eigen::VectorXd dt_f(6);
    dt_f << dx1, dy1, dz1, dx2, dy2, dz2;
    return dt_f;
}

// Eigen::MatrixXd CoupledRossler::jacobi