#include "openflex_isaac_controllers/explicit_topic_position_controller.hpp"

#include <algorithm>
#include <utility>

#include "pluginlib/class_list_macros.hpp"
#include "rclcpp/qos.hpp"

namespace openflex_isaac_controllers
{

controller_interface::CallbackReturn ExplicitTopicPositionController::on_init()
{
  try {
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("interface_name", hardware_interface::HW_IF_POSITION);
    auto_declare<std::string>("command_topic", "");
    joints_ = get_node()->get_parameter("joints").as_string_array();
    interface_name_ = get_node()->get_parameter("interface_name").as_string();
    command_topic_ = get_node()->get_parameter("command_topic").as_string();
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_node()->get_logger(), "Parameter declaration failed: %s", exception.what());
    return controller_interface::CallbackReturn::ERROR;
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
ExplicitTopicPositionController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto & joint : joints_) {
    configuration.names.push_back(joint + "/" + interface_name_);
  }
  return configuration;
}

controller_interface::InterfaceConfiguration
ExplicitTopicPositionController::state_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::NONE, {}};
}

controller_interface::CallbackReturn ExplicitTopicPositionController::on_configure(
  const rclcpp_lifecycle::State &)
{
  joints_ = get_node()->get_parameter("joints").as_string_array();
  interface_name_ = get_node()->get_parameter("interface_name").as_string();
  command_topic_ = get_node()->get_parameter("command_topic").as_string();

  if (joints_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "'joints' parameter was empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (interface_name_.empty()) {
    RCLCPP_ERROR(get_node()->get_logger(), "'interface_name' parameter was empty");
    return controller_interface::CallbackReturn::ERROR;
  }
  if (command_topic_.empty() || command_topic_.front() != '/') {
    RCLCPP_ERROR(
      get_node()->get_logger(), "'command_topic' must be an absolute ROS topic, got '%s'",
      command_topic_.c_str());
    return controller_interface::CallbackReturn::ERROR;
  }

  command_subscription_ = get_node()->create_subscription<Command>(
    command_topic_, rclcpp::QoS(10).best_effort(),
    [this](const std::shared_ptr<Command> message) { command_callback(std::move(message)); });
  RCLCPP_INFO(
    get_node()->get_logger(), "Explicit command controller: topic=%s joints=%zu interface=%s",
    command_topic_.c_str(), joints_.size(), interface_name_.c_str());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ExplicitTopicPositionController::on_activate(
  const rclcpp_lifecycle::State &)
{
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn ExplicitTopicPositionController::on_deactivate(
  const rclcpp_lifecycle::State &)
{
  command_buffer_.writeFromNonRT(std::shared_ptr<Command>());
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type ExplicitTopicPositionController::update(
  const rclcpp::Time &, const rclcpp::Duration &)
{
  const auto command = command_buffer_.readFromRT();
  if (command == nullptr || !(*command)) {
    return controller_interface::return_type::OK;
  }
  if ((*command)->data.size() != command_interfaces_.size()) {
    return controller_interface::return_type::ERROR;
  }
  for (std::size_t index = 0; index < command_interfaces_.size(); ++index) {
    command_interfaces_[index].set_value((*command)->data[index]);
  }
  return controller_interface::return_type::OK;
}

void ExplicitTopicPositionController::command_callback(const std::shared_ptr<Command> message)
{
  if (!message || message->data.size() != joints_.size()) {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 5000,
      "Ignoring command with %zu values; expected %zu", message ? message->data.size() : 0,
      joints_.size());
    return;
  }
  command_buffer_.writeFromNonRT(message);
}

}  // namespace openflex_isaac_controllers

PLUGINLIB_EXPORT_CLASS(
  openflex_isaac_controllers::ExplicitTopicPositionController,
  controller_interface::ControllerInterface)
