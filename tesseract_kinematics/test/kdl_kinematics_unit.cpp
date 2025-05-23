#include <tesseract_common/macros.h>
TESSERACT_COMMON_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <fstream>
TESSERACT_COMMON_IGNORE_WARNINGS_POP

#include "kinematics_test_utils.h"

#include <tesseract_kinematics/kdl/kdl_fwd_kin_chain.h>
#include <tesseract_kinematics/kdl/kdl_inv_kin_chain_lma.h>
#include <tesseract_kinematics/kdl/kdl_inv_kin_chain_nr.h>
#include <tesseract_kinematics/kdl/kdl_inv_kin_chain_nr_jl.h>

using namespace tesseract_kinematics::test_suite;

// TEST(TesseractKinematicsUnit, KDLKinChainLMAInverseKinematicUnit)  // NOLINT
// {
//   auto scene_graph = getSceneGraphIIWA();

//   tesseract_kinematics::KDLInvKinChainLMA::Config config;
//   tesseract_kinematics::KDLInvKinChainLMA derived_kin(*scene_graph, "base_link", "tool0", config);

//   tesseract_kinematics::KinematicsPluginFactory factory;
//   runInvKinIIWATest(factory, "KDLInvKinChainLMAFactory", "KDLFwdKinChainFactory");
// }

// TEST(TesseractKinematicsUnit, KDLKinChainNRInverseKinematicUnit)  // NOLINT
// {
//   auto scene_graph = getSceneGraphIIWA();

//   tesseract_kinematics::KDLInvKinChainNR::Config config;
//   tesseract_kinematics::KDLInvKinChainNR derived_kin(*scene_graph, "base_link", "tool0", config);

//   tesseract_kinematics::KinematicsPluginFactory factory;
//   runInvKinIIWATest(factory, "KDLInvKinChainNRFactory", "KDLFwdKinChainFactory");
// }

// TEST(TesseractKinematicsUnit, KDLKinChainNR_JLInverseKinematicUnit)  // NOLINT
// {
//   auto scene_graph = getSceneGraphIIWA();

//   tesseract_kinematics::KDLInvKinChainNR_JL::Config config;
//   tesseract_kinematics::KDLInvKinChainNR_JL derived_kin(*scene_graph, "base_link", "tool0", config);

//   tesseract_kinematics::KinematicsPluginFactory factory;
//   runInvKinIIWATest(factory, "KDLInvKinChainNR_JLFactory", "KDLFwdKinChainFactory");
// }

// TEST(TesseractKinematicsUnit, KDLKinChainNR_JL_AuboInverseKinematicUnit)  // NOLINT
// {
//   auto scene_graph = getSceneGraphAuboI5();

//   tesseract_kinematics::KDLInvKinChainNR_JL::Config config;
//   tesseract_kinematics::KDLInvKinChainNR_JL derived_kin(*scene_graph, "base_link", "wrist3_Link", config);

//   tesseract_kinematics::KinematicsPluginFactory factory;
//   runInvKinAuboTest(factory, "KDLInvKinChainNR_JLFactory", "KDLFwdKinChainFactory");
// }

TEST(TesseractKinematicsUnit, KDLKinChainNR_JL_JakaInverseKinematicUnit)  // NOLINT
{
  auto scene_graph = getSceneGraphJakaA12();

  tesseract_kinematics::KDLInvKinChainNR_JL::Config config;
  tesseract_kinematics::KDLInvKinChainNR_JL derived_kin(*scene_graph, "base_link", "J6", config);

  tesseract_kinematics::KinematicsPluginFactory factory;
  runInvKinJakaTest(factory, "KDLInvKinChainNR_JLFactory", "KDLFwdKinChainFactory");
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
