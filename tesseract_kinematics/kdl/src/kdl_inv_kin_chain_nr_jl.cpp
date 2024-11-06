/**
 * @file kdl_fwd_kin_chain_nr_jl.cpp
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
#include <tesseract_common/macros.h>
#include <tesseract_common/utils.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <console_bridge/console.h>
#include <tesseract_scene_graph/graph.h>
#include <tesseract_scene_graph/kdl_parser.h>
#include <memory>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include <tesseract_kinematics/kdl/kdl_inv_kin_chain_nr_jl.h>

#include <limits>
#include <chrono>
#include <iostream>

using Clock = std::chrono::steady_clock;
using namespace KDL;

namespace tesseract_kinematics
{
using Eigen::MatrixXd;
using Eigen::VectorXd;

KDLInvKinChainNR_JL::KDLInvKinChainNR_JL(const tesseract_scene_graph::SceneGraph& scene_graph,
                                         const std::vector<std::pair<std::string, std::string>>& chains,
                                         Config kdl_config,
                                         std::string solver_name)
  : kdl_config_(kdl_config), solver_name_(std::move(solver_name))
{
  if (!scene_graph.getLink(scene_graph.getRoot()))
    throw std::runtime_error("The scene graph has an invalid root.");

  if (!parseSceneGraph(kdl_data_, scene_graph, chains))
    throw std::runtime_error("Failed to parse KDL data from Scene Graph");

  // Create KDL FK and IK Solver
  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_data_.robot_chain);
  ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(
      kdl_data_.robot_chain, kdl_config_.vel_eps, kdl_config_.vel_iterations);
  ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(kdl_data_.robot_chain,
                                                             kdl_data_.q_min,
                                                             kdl_data_.q_max,
                                                             *fk_solver_,
                                                             *ik_vel_solver_,
                                                             kdl_config_.pos_iterations,
                                                             kdl_config_.pos_eps);

  // parse types
  auto chain = kdl_data_.robot_chain;
  for (unsigned int i = 0; i < chain.segments.size(); i++)
  {
    std::string type = chain.segments[i].getJoint().getTypeName();
    if (type.find("Rot") != std::string::npos)
    {
      if (kdl_data_.q_max(static_cast<unsigned int>(types.size())) >= std::numeric_limits<float>::max() &&
          kdl_data_.q_min(static_cast<unsigned int>(types.size())) <= std::numeric_limits<float>::lowest())
        types.push_back(tesseract_kinematics::BasicJointType::Continuous);
      else types.push_back(tesseract_kinematics::BasicJointType::RotJoint);
    }
    else if (type.find("Trans") != std::string::npos) {
      types.push_back(tesseract_kinematics::BasicJointType::TransJoint);
    }
  }
}

KDLInvKinChainNR_JL::KDLInvKinChainNR_JL(const tesseract_scene_graph::SceneGraph& scene_graph,
                                         const std::string& base_link,
                                         const std::string& tip_link,
                                         Config kdl_config,
                                         std::string solver_name)
  : KDLInvKinChainNR_JL(scene_graph, { std::make_pair(base_link, tip_link) }, kdl_config, std::move(solver_name))
{
}

InverseKinematics::UPtr KDLInvKinChainNR_JL::clone() const { return std::make_unique<KDLInvKinChainNR_JL>(*this); }

KDLInvKinChainNR_JL::KDLInvKinChainNR_JL(const KDLInvKinChainNR_JL& other) { *this = other; }

KDLInvKinChainNR_JL& KDLInvKinChainNR_JL::operator=(const KDLInvKinChainNR_JL& other)
{
  kdl_data_ = other.kdl_data_;
  kdl_config_ = other.kdl_config_;
  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(kdl_data_.robot_chain);
  ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(
      kdl_data_.robot_chain, kdl_config_.vel_eps, kdl_config_.vel_iterations);
  ik_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(kdl_data_.robot_chain,
                                                             kdl_data_.q_min,
                                                             kdl_data_.q_max,
                                                             *fk_solver_,
                                                             *ik_vel_solver_,
                                                             kdl_config_.pos_iterations,
                                                             kdl_config_.pos_eps);
  solver_name_ = other.solver_name_;
  types = other.types;

  return *this;
}

IKSolutions KDLInvKinChainNR_JL::calcInvKinHelper(const Eigen::Isometry3d& pose,
                                                  const Eigen::Ref<const Eigen::VectorXd>& seed,
                                                  int /*segment_num*/) const
{
  assert(std::abs(1.0 - pose.matrix().determinant()) < 1e-6);  // NOLINT

  int run_cnt = 0;
  unsigned int n = kdl_data_.robot_chain.getNrOfJoints();
  auto start_time = Clock::now();
  double time_left;
  KDL::Frame f;
  KDL::Twist delta_twist;
  KDL::JntArray delta_q(n);

  KDL::Frame p_in;
  EigenToKDL(pose, p_in);
  KDL::JntArray q_out;
  EigenToKDL(seed, q_out);

  // printJntArray(q_out);
  // printFrame(p_in);

  Eigen::VectorXd solution(seed.size());
  auto q_min =  kdl_data_.q_min;
  auto q_max =  kdl_data_.q_max;

  IKSolutions solutions;
  double sol_max_diff = tesseract_common::getMaxTimeFromEnv("sol_max_diff", 0.01);
  // 定义一个用于检查解是否重复的lambda函数
  auto isSolutionUnique = [&](const Eigen::VectorXd& sol) {
    for (const auto& existing_sol : solutions)
    {
      if ((existing_sol - sol).norm() < sol_max_diff)  // 假设阈值为1e-6
      {
        return false;  // 发现重复
      }
    }
    return true;  // 没有发现重复
  };

  double KDL_IK_max_time = tesseract_common::getMaxTimeFromEnv("KDL_IK_max_time", 0.1);

  do
  {
    fk_solver_->JntToCart(q_out, f);
    delta_twist = diffRelative(p_in, f);
    run_cnt = run_cnt + 1;

    if (std::abs(delta_twist.vel.x()) <= std::abs(bounds.vel.x())) delta_twist.vel.x(0);
    if (std::abs(delta_twist.vel.y()) <= std::abs(bounds.vel.y())) delta_twist.vel.y(0);
    if (std::abs(delta_twist.vel.z()) <= std::abs(bounds.vel.z())) delta_twist.vel.z(0);
    if (std::abs(delta_twist.rot.x()) <= std::abs(bounds.rot.x())) delta_twist.rot.x(0);
    if (std::abs(delta_twist.rot.y()) <= std::abs(bounds.rot.y())) delta_twist.rot.y(0);
    if (std::abs(delta_twist.rot.z()) <= std::abs(bounds.rot.z())) delta_twist.rot.z(0);

    if (Equal(delta_twist, KDL::Twist::Zero(), eps))
    {
      // std::cout << "找到一个解，运行次数: " << run_cnt << std::endl;
      // printJntArray(q_out);
      // std::cout << "calcInvKinHelper pose: \n" << pose.matrix() << std::endl;
      KDLToEigen(q_out, solution);
      
      // 检查解是否唯一
      if (isSolutionUnique(solution))
      {
        solutions.push_back(solution);  // 仅在解唯一时添加
      }
    }

    delta_twist = diff(f, p_in);
    ik_vel_solver_->CartToJnt(q_out, delta_twist, delta_q);
    KDL::JntArray q_curr(n);
    KDL::Add(q_out, delta_q, q_curr);

    for (unsigned int j = 0; j < q_min.data.size(); j++)
    {
      if (types[j] == tesseract_kinematics::BasicJointType::Continuous)
        continue;
      if (q_curr(j) < q_min(j))
      {
        if (!wrap || types[j] == tesseract_kinematics::BasicJointType::TransJoint)
          q_curr(j) = q_min(j);
        else
        {
          double diffangle = fmod(q_min(j) - q_curr(j), 2 * M_PI);
          double curr_angle = q_min(j) - diffangle + 2 * M_PI;
          q_curr(j) = (curr_angle > q_max(j)) ? q_min(j) : curr_angle;
        }
      }
    }

    for (unsigned int j = 0; j < q_max.data.size(); j++)
    {
      if (types[j] == tesseract_kinematics::BasicJointType::Continuous)
        continue;

      if (q_curr(j) > q_max(j))
      {
        if (!wrap || types[j] == tesseract_kinematics::BasicJointType::TransJoint)
          q_curr(j) = q_max(j);
        else
        {
          double diffangle = fmod(q_curr(j) - q_max(j), 2 * M_PI);
          double curr_angle = q_max(j) + diffangle - 2 * M_PI;
          q_curr(j) = (curr_angle < q_min(j)) ? q_max(j) : curr_angle;
        }
      }
    }

    KDL::Subtract(q_out, q_curr, q_out);

    if (q_out.data.isZero(std::numeric_limits<float>::epsilon()))
    {
      if (rr)
      {
        for (unsigned int j = 0; j < q_out.data.size(); j++)
        {
          if (types[j] == tesseract_kinematics::BasicJointType::Continuous)
            q_curr(j) = fRand(q_curr(j) - 2 * M_PI, q_curr(j) + 2 * M_PI);
          else
            q_curr(j) = fRand(q_min(j), q_max(j));
        }
      }
    }

    q_out = q_curr;
    auto timediff = Clock::now() - start_time;
    time_left = KDL_IK_max_time - std::chrono::duration<double>(timediff).count();
  } while (time_left > 0);

  printFrame(p_in);
  // 在返回前打印找到的解的数量
  std::cout << "Total number of solutions found: " << solutions.size() << std::endl;

  std::cout << "Total number of runs: " << run_cnt << std::endl;
  for (const auto& sol : solutions)
  {
    std::cout << "found one solution:\t" << sol.transpose() << std::endl;
  }
  return solutions;  // 返回去重后的解
}

IKSolutions KDLInvKinChainNR_JL::calcInvKin(const tesseract_common::TransformMap& tip_link_poses,
                                            const Eigen::Ref<const Eigen::VectorXd>& seed) const
{
  assert(tip_link_poses.find(kdl_data_.tip_link_name) != tip_link_poses.end());
  return calcInvKinHelper(tip_link_poses.at(kdl_data_.tip_link_name), seed);
}

std::vector<std::string> KDLInvKinChainNR_JL::getJointNames() const { return kdl_data_.joint_names; }

Eigen::Index KDLInvKinChainNR_JL::numJoints() const { return kdl_data_.robot_chain.getNrOfJoints(); }

std::string KDLInvKinChainNR_JL::getBaseLinkName() const { return kdl_data_.base_link_name; }

std::string KDLInvKinChainNR_JL::getWorkingFrame() const { return kdl_data_.base_link_name; }

std::vector<std::string> KDLInvKinChainNR_JL::getTipLinkNames() const { return { kdl_data_.tip_link_name }; }

std::string KDLInvKinChainNR_JL::getSolverName() const { return solver_name_; }

}  // namespace tesseract_kinematics
