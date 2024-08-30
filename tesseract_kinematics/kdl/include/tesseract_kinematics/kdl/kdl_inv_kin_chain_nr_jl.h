/**
 * @file kdl_fwd_kin_chain_nr_jl.h
 * @brief Tesseract KDL inverse kinematics chain Newton-Raphson implementation.
 *
 * @author Levi Armstrong, Roelof Oomen
 * @date July 26, 2023
 * @version TODO
 * @bug No known bugs
 *
 * @copyright Copyright (c) 2023, Southwest Research Institute
 *
 * @par License
 * Software License Agreement (Apache License)
 * @par
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * @par
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TESSERACT_KINEMATICS_KDL_INV_KIN_CHAIN_NR_JL_H
#define TESSERACT_KINEMATICS_KDL_INV_KIN_CHAIN_NR_JL_H
#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <mutex>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <tesseract_kinematics/core/inverse_kinematics.h>
#include <tesseract_kinematics/kdl/kdl_utils.h>
#include <iostream>

namespace tesseract_kinematics
{
static const std::string KDL_INV_KIN_CHAIN_NR_JL_SOLVER_NAME = "KDLInvKinChainNR_JL";
enum BasicJointType { RotJoint, TransJoint, Continuous };

/**
 * @brief KDL Inverse kinematic chain implementation.
 */
class KDLInvKinChainNR_JL : public InverseKinematics
{
public:
  // LCOV_EXCL_START
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // LCOV_EXCL_STOP

  using Ptr = std::shared_ptr<KDLInvKinChainNR_JL>;
  using ConstPtr = std::shared_ptr<const KDLInvKinChainNR_JL>;
  using UPtr = std::unique_ptr<KDLInvKinChainNR_JL>;
  using ConstUPtr = std::unique_ptr<const KDLInvKinChainNR_JL>;

  /**
   * @brief The Config struct
   *
   * This contains parameters that can be used to customize the KDL solver for your application.
   * They are ultimately passed to the constuctors of the underlying ChainIkSolver.
   * The NR version creates both position and velocity solvers with different defaults for each.
   *
   * The defaults provided here are the same defaults imposed by the KDL library.
   */
  struct Config
  {
    double vel_eps{ 0.00001 };
    int vel_iterations{ 150 };
    double pos_eps{ 1e-6 };
    int pos_iterations{ 100 };
  };

  ~KDLInvKinChainNR_JL() override = default;
  KDLInvKinChainNR_JL(const KDLInvKinChainNR_JL& other);
  KDLInvKinChainNR_JL& operator=(const KDLInvKinChainNR_JL& other);
  KDLInvKinChainNR_JL(KDLInvKinChainNR_JL&&) = delete;
  KDLInvKinChainNR_JL& operator=(KDLInvKinChainNR_JL&&) = delete;

  /**
   * @brief Construct KDL Forward Kinematics
   * Creates KDL::Chain from tesseract scene graph
   * @param scene_graph The Tesseract Scene Graph
   * @param base_link The name of the base link for the kinematic chain
   * @param tip_link The name of the tip link for the kinematic chain
   * @param solver_name The solver name of the kinematic chain
   */
  KDLInvKinChainNR_JL(const tesseract_scene_graph::SceneGraph& scene_graph,
                      const std::string& base_link,
                      const std::string& tip_link,
                      Config kdl_config,
                      std::string solver_name = KDL_INV_KIN_CHAIN_NR_JL_SOLVER_NAME);

  /**
   * @brief Construct Inverse Kinematics as chain
   * Creates a inverse kinematic chain object from sequential chains
   * @param scene_graph The Tesseract Scene Graph
   * @param chains A vector of kinematics chains <base_link, tip_link> that get concatenated
   * @param solver_name The solver name of the kinematic chain
   */
  KDLInvKinChainNR_JL(const tesseract_scene_graph::SceneGraph& scene_graph,
                      const std::vector<std::pair<std::string, std::string> >& chains,
                      Config kdl_config,
                      std::string solver_name = KDL_INV_KIN_CHAIN_NR_JL_SOLVER_NAME);

  IKSolutions calcInvKin(const tesseract_common::TransformMap& tip_link_poses,
                         const Eigen::Ref<const Eigen::VectorXd>& seed) const override final;

  std::vector<std::string> getJointNames() const override final;
  Eigen::Index numJoints() const override final;
  std::string getBaseLinkName() const override final;
  std::string getWorkingFrame() const override final;
  std::vector<std::string> getTipLinkNames() const override final;
  std::string getSolverName() const override final;
  InverseKinematics::UPtr clone() const override final;

  inline void setMaxtime(double t)
  {
    maxtime = t;
  }

private:
  KDLChainData kdl_data_;                                      /**< @brief KDL data parsed from Scene Graph */
  Config kdl_config_;                                          /**< @brief KDL configuration data parsed from YAML */
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_; /**< @brief KDL Forward Kinematic Solver */
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_vel_solver_;  /**< @brief KDL Inverse kinematic velocity solver */
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_solver_;     /**< @brief KDL Inverse kinematic solver */
  std::string solver_name_{ KDL_INV_KIN_CHAIN_NR_JL_SOLVER_NAME }; /**< @brief Name of this solver */
  mutable std::mutex mutex_; /**< @brief KDL is not thread safe due to mutable variables in Joint Class */
  // KDL::Twist bounds = KDL::Twist::Zero();
  KDL::Twist bounds = KDL::Twist(KDL::Vector(1e-5, 1e-5, 1e-5), KDL::Vector(1e-3, 1e-3, 1e-3));
  double maxtime = 10;
  double eps = 1e-6;

  bool rr = true;
  bool wrap = true;

  std::vector<tesseract_kinematics::BasicJointType> types;

  inline static double fRand(double min, double max)
  {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
  }

  /** @brief calcFwdKin helper function */
  IKSolutions calcInvKinHelper(const Eigen::Isometry3d& pose,
                               const Eigen::Ref<const Eigen::VectorXd>& seed,
                               int segment_num = -1) const;
};

}  // namespace tesseract_kinematics

/**
 * determines the rotation axis necessary to rotate from frame b1 to the
 * orientation of frame b2 and the vector necessary to translate the origin
 * of b1 to the origin of b2, and stores the result in a Twist
 * datastructure.  The result is w.r.t. frame b1.
 * \param F_a_b1 frame b1 expressed with respect to some frame a.
 * \param F_a_b2 frame b2 expressed with respect to some frame a.
 * \warning The result is not a real Twist!
 * \warning In contrast to standard KDL diff methods, the result of
 * diffRelative is w.r.t. frame b1 instead of frame a.
 */
IMETHOD KDL::Twist diffRelative(const KDL::Frame & F_a_b1, const KDL::Frame & F_a_b2, double dt = 1)
{
  return KDL::Twist(F_a_b1.M.Inverse() * diff(F_a_b1.p, F_a_b2.p, dt),
               F_a_b1.M.Inverse() * diff(F_a_b1.M, F_a_b2.M, dt));
}

IMETHOD void printJntArray(const KDL::JntArray& jntArray) {
    std::cout << "JntArray: [";
    for (unsigned int i = 0; i < jntArray.rows(); ++i) {
        std::cout << jntArray(i);
        if (i < jntArray.rows() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}


IMETHOD void printTwist(const KDL::Twist& twist) {
    std::cout << "Twist:" << std::endl;
    std::cout << "  Linear velocity: [" 
              << twist.vel.x() << ", "
              << twist.vel.y() << ", "
              << twist.vel.z() << "]" << std::endl;
    std::cout << "  Angular velocity: ["
              << twist.rot.x() << ", "
              << twist.rot.y() << ", "
              << twist.rot.z() << "]" << std::endl;
}

IMETHOD void printFrame(const KDL::Frame& frame) {
    // 打印旋转矩阵
    KDL::Rotation R = frame.M;
    double  x, y, z,  w;
    R.GetQuaternion(x, y, z, w);
    std::cout << "quaternion：" << "x: " << x << " y: " << y << " z: " << z << " w: " << w << std::endl;
    // 打印平移向量
    KDL::Vector p = frame.p;
    std::cout << "Translation vector: [" 
              << p.x() << ", " 
              << p.y() << ", " 
              << p.z() << "]" << std::endl;
}


#endif  // TESSERACT_KINEMATICS_KDL_INV_KIN_CHAIN_NR_JL_H
