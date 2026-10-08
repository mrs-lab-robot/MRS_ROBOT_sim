#pragma once

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "rclcpp/subscription.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

namespace openflex_isaac_controllers
{

class ExplicitTopicPositionController : public controller_interface::ControllerInterface
{
public:
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_init() override;
  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  using Command = std_msgs::msg::Float64MultiArray;

  void command_callback(const std::shared_ptr<Command> message);

  std::vector<std::string> joints_;
  std::string interface_name_;
  std::string command_topic_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<Command>> command_buffer_;
  rclcpp::Subscription<Command>::SharedPtr command_subscription_;
};

}  // namespace openflex_isaac_controllers
