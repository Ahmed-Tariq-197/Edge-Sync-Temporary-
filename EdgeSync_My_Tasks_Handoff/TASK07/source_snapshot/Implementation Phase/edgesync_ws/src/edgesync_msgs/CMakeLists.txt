cmake_minimum_required(VERSION 3.8)
project(edgesync_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/HandoffHeader.msg"
  "msg/HandoffRequest.msg"
  "msg/HandoffResponse.msg"
  "msg/HandoffCancel.msg"
  "msg/AcqStatus.msg"
  "msg/HandoffConfirm.msg"
  "msg/HandoffConfirmAck.msg"
  "msg/OwnershipEvent.msg"
  "msg/StateHeartbeat.msg"
  "msg/Target3DState.msg"
  "msg/TTOEstimate.msg"
  "msg/ConsensusBelief.msg"
  # ES-EXP07 <-> ES-EXP13: carrier for the reference Re-ID embedding a handoff request must carry.
  # Additive registration only; no existing type above was edited or reordered.
  "msg/ReidSignature.msg"
)

set(srv_files
  "srv/IdentityVerify.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES
    builtin_interfaces
    sensor_msgs
    geometry_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
