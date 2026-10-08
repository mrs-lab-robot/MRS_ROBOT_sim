#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QImage>
#include <QLabel>
#include <QPointF>
#include <QTabWidget>
#include <QWidget>

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "lift_slide_panel/lift_panel.hpp"
#include "swerve_base_panel/swerve_base_panel.hpp"

namespace openflex_isaac_bringup
{

class RadarView;

class SimulationControlSensorsPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit SimulationControlSensorsPanel(QWidget * parent = nullptr);
  ~SimulationControlSensorsPanel() override = default;

  void onInitialize() override;
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private:
  void buildSensorPage();
  void subscribeToSelectedSensors();
  void showCameraFrame(const std::string & key, const QImage & image);
  void showRadarData(const std::string & key, std::vector<QPointF> points);
  void setRadarStatus(const QString & status);

  QTabWidget * page_tabs_{nullptr};
  QTabWidget * sensor_tabs_{nullptr};
  QTabWidget * camera_tabs_{nullptr};
  QTabWidget * radar_tabs_{nullptr};
  QWidget * sensor_page_{nullptr};
  QWidget * camera_page_{nullptr};
  QWidget * radar_page_{nullptr};
  QLabel * camera_placeholder_{nullptr};
  QLabel * radar_placeholder_{nullptr};
  swerve_base_panel::SwerveBasePanel * base_panel_{nullptr};
  lift_slide_panel::LiftPanel * lift_panel_{nullptr};
  std::map<std::string, QLabel *> camera_image_labels_;
  std::map<std::string, RadarView *> radar_views_;
  std::map<std::string, std::string> camera_topics_;
  std::map<std::string, std::string> camera_labels_;
  rclcpp::Node::SharedPtr node_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
    camera_subscriptions_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_subscription_;
  bool rviz_initialized_{false};
  bool show_cameras_{false};
  bool show_lidar_{false};
};

}  // namespace openflex_isaac_bringup
