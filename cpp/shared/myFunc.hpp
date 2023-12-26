#include <chrono>
#include <eigen3/Eigen/Dense>

namespace myfunc{
    void duration(std::chrono::time_point<std::chrono::system_clock> start);
    int shift(double pre_theta, double theta, int rotation_number);
    Eigen::VectorXd rungeKuttaJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian, double dt);
    Eigen::VectorXd computeDerivativeJacobian(const Eigen::VectorXd& state, const Eigen::MatrixXd& jacobian);
    Eigen::MatrixXd loc_max(const Eigen::MatrixXd& traj, int loc_max_dim);
}
