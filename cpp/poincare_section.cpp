#include <iostream>
#include <iomanip>
#include <eigen3/Eigen/Dense>
#include <complex>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include "shared/myFunc.hpp"
#include "shared/Flow.hpp"
#include "shared/Map.hpp"
#include "shared/matplotlibcpp.h"
#include "shared/Eigen_numpy_converter.hpp"
namespace plt = matplotlibcpp;

int main(){
    auto start = std::chrono::system_clock::now(); // timer start
    double dt = 0.01;
    double t_0 = 0;
    double t = 1e+8;
    double dump = 1e+4;
    CRparams params;
    params.omega1 = 0.95;
    params.omega2 = 0.99;
    params.epsilon = 0.03;
    params.a = 0.165;
    params.c = 10;
    params.f = 0.2;
    Eigen::VectorXd x_0 = (Eigen::VectorXd::Random(6).array()) * 10;
    // Eigen::VectorXd x_0 = npy2EigenVec<double>("../initials/chaotic.npy", true);
    int plot_dim1 = 3;
    int plot_dim2 = 4;
    CoupledRossler CR(params, dt, t_0, t, dump, x_0);
    Eigen::MatrixXd trajectory = CR.get_trajectory();
    
    PoincareMap PM(trajectory);
    PM.locmax(0);
    Eigen::MatrixXd section = PM.get();

    std::cout << section.size() <<"points"<< std::endl; //print the number of points
    std::map<std::string, std::string> plotSettings;
    plotSettings["font.family"] = "Times New Roman";
    plotSettings["font.size"] = "10";
    plt::rcparams(plotSettings);
    plt::figure_size(1200, 1200);
    // Add graph title
    std::vector<double> x(section.size()),y(section.size());
    for(int i = 0; i < section.size(); i++){
        x[i] = section(i, plot_dim1 - 1);
        y[i] = section(i, plot_dim2 - 1);
    }
    plt::scatter(x,y,5.0);
    std::ostringstream oss;
    oss <<"Shell"<< plot_dim1;
    plt::xlabel(oss.str()); 
    oss.str("");
    oss <<"Shell"<< plot_dim2;
    plt::ylabel(oss.str());
    plt::ylim(0.15, 0.4);
    plt::xlim(0.05, 0.6);
    oss.str("");
    oss << "../../poincare/beta_" << beta << "nu_" << nu << "loc_max_4"<< t / latter <<"period.png";  // 文字列を結合する
    // oss << "../../poincare/beta_" << beta << "nu_" << nu << "loc_max_4_laminar50000period.png";
    std::string plotfname = oss.str(); // 文字列を取得する
    std::cout << "Saving result to " << plotfname << std::endl;
    plt::save(plotfname);

    myfunc::duration(start);
}
