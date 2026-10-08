#include "openflex_isaac_bringup/simulation_control_sensors_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

#include <QColor>
#include <QMetaObject>
#include <QPainter>
#include <QPaintEvent>
#include <QPixmap>
#include <QResizeEvent>
#include <QSizePolicy>
#include <QVBoxLayout>

#include <rviz_common/display_context.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <sensor_msgs/msg/point_field.hpp>

namespace openflex_isaac_bringup
{
namespace
{

struct CameraSpec
{
  const char * key;
  const char * label;
  const char * topic;
};

constexpr CameraSpec kCameraSpecs[] = {
  {"head", "头部相机", "/cam_head/color/image"},
  {"left", "左手相机", "/cam_left/color/image"},
  {"right", "右手相机", "/cam_right/color/image"},
  {"base", "底盘相机", "/cam_base/color/image"},
};

QImage imageFromRosMessage(const sensor_msgs::msg::Image & message)
{
  if (message.width == 0 || message.height == 0 || message.step == 0 ||
    message.data.size() < static_cast<std::size_t>(message.step) * message.height)
  {
    return {};
  }

  QImage::Format format = QImage::Format_Invalid;
  if (message.encoding == "rgb8") {
    format = QImage::Format_RGB888;
  } else if (message.encoding == "bgr8") {
    format = QImage::Format_BGR888;
  } else if (message.encoding == "rgba8") {
    format = QImage::Format_RGBA8888;
  } else if (message.encoding == "bgra8") {
    format = QImage::Format_ARGB32;
  } else if (message.encoding == "mono8") {
    format = QImage::Format_Grayscale8;
  }
  if (format == QImage::Format_Invalid) {
    return {};
  }

  return QImage(
    message.data.data(), static_cast<int>(message.width),
    static_cast<int>(message.height), static_cast<int>(message.step), format).copy();
}

const sensor_msgs::msg::PointField * findField(
  const sensor_msgs::msg::PointCloud2 & message, const std::string & name)
{
  const auto it = std::find_if(
    message.fields.begin(), message.fields.end(), [&name](const auto & field) {
      return field.name == name && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
    });
  return it == message.fields.end() ? nullptr : &(*it);
}

std::vector<QPointF> pointsFromCloud(const sensor_msgs::msg::PointCloud2 & message)
{
  const auto * x_field = findField(message, "x");
  const auto * y_field = findField(message, "y");
  if (x_field == nullptr || y_field == nullptr || message.point_step == 0 ||
    message.width == 0 || x_field->offset + sizeof(float) > message.point_step ||
    y_field->offset + sizeof(float) > message.point_step)
  {
    return {};
  }

  const std::size_t count = static_cast<std::size_t>(message.width) * message.height;
  const std::size_t stride = std::max<std::size_t>(1, count / 6000);
  std::vector<QPointF> points;
  points.reserve(std::min<std::size_t>(count / stride + 1, 6001));
  for (std::size_t index = 0; index < count; index += stride) {
    const std::size_t row = index / message.width;
    const std::size_t column = index % message.width;
    const std::size_t offset = row * message.row_step + column * message.point_step;
    if (offset + std::max(x_field->offset, y_field->offset) + sizeof(float) >
      message.data.size())
    {
      break;
    }
    float x = 0.0F;
    float y = 0.0F;
    std::memcpy(&x, message.data.data() + offset + x_field->offset, sizeof(float));
    std::memcpy(&y, message.data.data() + offset + y_field->offset, sizeof(float));
    if (std::isfinite(x) && std::isfinite(y)) {
      points.emplace_back(x, y);
    }
  }
  return points;
}

std::vector<QPointF> pointsFromScan(const sensor_msgs::msg::LaserScan & message)
{
  const std::size_t stride = std::max<std::size_t>(1, message.ranges.size() / 2400);
  std::vector<QPointF> points;
  points.reserve(message.ranges.size() / stride + 1);
  for (std::size_t index = 0; index < message.ranges.size(); index += stride) {
    const float range = message.ranges[index];
    if (!std::isfinite(range) || range < message.range_min || range > message.range_max) {
      continue;
    }
    const double angle = message.angle_min + static_cast<double>(index) * message.angle_increment;
    points.emplace_back(range * std::cos(angle), range * std::sin(angle));
  }
  return points;
}

class CameraImageLabel : public QLabel
{
public:
  explicit CameraImageLabel(QWidget * parent = nullptr)
  : QLabel(parent)
  {
    setAlignment(Qt::AlignCenter);
    setMinimumSize(320, 180);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    setStyleSheet("background:#101827;color:#dbe5f2;");
  }

  void setImage(const QImage & image)
  {
    image_ = image;
    updatePixmap();
  }

protected:
  void resizeEvent(QResizeEvent * event) override
  {
    QLabel::resizeEvent(event);
    updatePixmap();
  }

private:
  void updatePixmap()
  {
    if (!image_.isNull() && width() > 0 && height() > 0) {
      setPixmap(QPixmap::fromImage(image_).scaled(
        size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    }
  }

  QImage image_;
};

}  // namespace

class RadarView : public QWidget
{
public:
  explicit RadarView(QWidget * parent = nullptr)
  : QWidget(parent)
  {
    setMinimumSize(300, 220);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
  }

  void setPoints(std::vector<QPointF> points)
  {
    points_ = std::move(points);
    update();
  }

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);
    painter.fillRect(rect(), QColor("#101827"));

    const QPointF center(width() / 2.0, height() / 2.0);
    const double radius = std::max(10.0, std::min(width(), height()) * 0.43);
    constexpr double maximum_range = 12.0;
    painter.setPen(QPen(QColor("#41516a"), 1));
    for (double range : {2.0, 4.0, 8.0, 12.0}) {
      const double ring_radius = radius * range / maximum_range;
      painter.drawEllipse(center, ring_radius, ring_radius);
      painter.setPen(QColor("#9aaac0"));
      painter.drawText(
        QPointF(center.x() + 3.0, center.y() - ring_radius - 3.0),
        QString::number(range, 'f', 0) + QStringLiteral(" m"));
      painter.setPen(QPen(QColor("#41516a"), 1));
    }
    painter.drawLine(QPointF(center.x() - radius, center.y()),
      QPointF(center.x() + radius, center.y()));
    painter.drawLine(QPointF(center.x(), center.y() - radius),
      QPointF(center.x(), center.y() + radius));

    painter.setPen(Qt::NoPen);
    painter.setBrush(QColor("#53d3a5"));
    for (const auto & point : points_) {
      const double distance = std::hypot(point.x(), point.y());
      if (distance > maximum_range) {
        continue;
      }
      const QPointF screen_point(
        center.x() - point.y() * radius / maximum_range,
        center.y() - point.x() * radius / maximum_range);
      painter.drawEllipse(screen_point, 2.2, 2.2);
    }
  }

private:
  std::vector<QPointF> points_;
};

SimulationControlSensorsPanel::SimulationControlSensorsPanel(QWidget * parent)
: rviz_common::Panel(parent)
{
  auto * root = new QVBoxLayout(this);
  root->setContentsMargins(4, 4, 4, 4);
  root->setSpacing(4);

  page_tabs_ = new QTabWidget(this);
  page_tabs_->setObjectName(QStringLiteral("SimulationControlSensorPages"));
  page_tabs_->setDocumentMode(true);
  auto * control_page = new QWidget(page_tabs_);
  auto * control_layout = new QVBoxLayout(control_page);
  control_layout->setContentsMargins(2, 2, 2, 2);
  control_layout->setSpacing(4);
  base_panel_ = new swerve_base_panel::SwerveBasePanel(control_page);
  base_panel_->setObjectName(QStringLiteral("SimulationSwerveBasePanel"));
  lift_panel_ = new lift_slide_panel::LiftPanel(control_page);
  lift_panel_->setObjectName(QStringLiteral("SimulationLiftPanel"));
  control_layout->addWidget(base_panel_, 2);
  control_layout->addWidget(lift_panel_, 3);
  page_tabs_->addTab(control_page, QStringLiteral("底盘+升降控制"));

  sensor_page_ = new QWidget(page_tabs_);
  auto * sensor_layout = new QVBoxLayout(sensor_page_);
  sensor_layout->setContentsMargins(2, 2, 2, 2);
  sensor_tabs_ = new QTabWidget(sensor_page_);
  sensor_tabs_->setObjectName(QStringLiteral("SimulationSensorPages"));
  sensor_layout->addWidget(sensor_tabs_);

  camera_page_ = new QWidget(sensor_tabs_);
  auto * camera_layout = new QVBoxLayout(camera_page_);
  camera_placeholder_ = new QLabel(QStringLiteral("等待已启动的相机发布图像…"), camera_page_);
  camera_placeholder_->setAlignment(Qt::AlignCenter);
  camera_placeholder_->setWordWrap(true);
  camera_placeholder_->setStyleSheet("color:#708096;padding:18px;");
  camera_layout->addWidget(camera_placeholder_);

  radar_page_ = new QWidget(sensor_tabs_);
  auto * radar_layout = new QVBoxLayout(radar_page_);
  radar_placeholder_ = new QLabel(
    QStringLiteral("等待 /livox/lidar_points 或 /scan 数据…"), radar_page_);
  radar_placeholder_->setObjectName(QStringLiteral("SimulationRadarStatus"));
  radar_placeholder_->setAlignment(Qt::AlignCenter);
  radar_placeholder_->setWordWrap(true);
  radar_tabs_ = new QTabWidget(radar_page_);
  radar_tabs_->setObjectName(QStringLiteral("SimulationRadarTopics"));
  radar_tabs_->hide();
  radar_layout->addWidget(radar_placeholder_);
  radar_layout->addWidget(radar_tabs_, 1);

  root->addWidget(page_tabs_);
}

void SimulationControlSensorsPanel::onInitialize()
{
  auto * context = getDisplayContext();
  base_panel_->initialize(context);
  lift_panel_->initialize(context);
  rviz_initialized_ = true;
  buildSensorPage();
  subscribeToSelectedSensors();
}

void SimulationControlSensorsPanel::buildSensorPage()
{
  if (!show_cameras_ && !show_lidar_) {
    return;
  }
  if (show_cameras_ && sensor_tabs_->indexOf(camera_page_) == -1) {
    sensor_tabs_->addTab(camera_page_, QStringLiteral("相机"));
  } else if (!show_cameras_ && sensor_tabs_->indexOf(camera_page_) != -1) {
    sensor_tabs_->removeTab(sensor_tabs_->indexOf(camera_page_));
  }
  if (show_lidar_ && sensor_tabs_->indexOf(radar_page_) == -1) {
    sensor_tabs_->addTab(radar_page_, QStringLiteral("雷达"));
  } else if (!show_lidar_ && sensor_tabs_->indexOf(radar_page_) != -1) {
    sensor_tabs_->removeTab(sensor_tabs_->indexOf(radar_page_));
  }
  if ((show_cameras_ || show_lidar_) && page_tabs_->indexOf(sensor_page_) == -1) {
    page_tabs_->addTab(sensor_page_, QStringLiteral("相机与雷达"));
  } else if (!show_cameras_ && !show_lidar_ && page_tabs_->indexOf(sensor_page_) != -1) {
    page_tabs_->removeTab(page_tabs_->indexOf(sensor_page_));
  }
}

void SimulationControlSensorsPanel::subscribeToSelectedSensors()
{
  const auto abstraction = getDisplayContext()->getRosNodeAbstraction().lock();
  if (!abstraction) {
    setRadarStatus(QStringLiteral("RViz ROS 节点不可用。"));
    return;
  }
  node_ = abstraction->get_raw_node();
  if (!node_) {
    setRadarStatus(QStringLiteral("RViz ROS 节点不可用。"));
    return;
  }

  if (show_cameras_ && camera_subscriptions_.empty()) {
    for (const auto & spec : kCameraSpecs) {
      const std::string key(spec.key);
      camera_topics_[key] = spec.topic;
      camera_labels_[key] = spec.label;
      camera_subscriptions_.push_back(node_->create_subscription<sensor_msgs::msg::Image>(
          spec.topic, rclcpp::SensorDataQoS(),
          [this, key](sensor_msgs::msg::Image::ConstSharedPtr message) {
            const QImage image = imageFromRosMessage(*message);
            if (image.isNull()) {
              return;
            }
            QMetaObject::invokeMethod(this, [this, key, image]() {
              showCameraFrame(key, image);
            }, Qt::QueuedConnection);
          }));
    }
  }

  if (show_lidar_ && !pointcloud_subscription_) {
    pointcloud_subscription_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/livox/lidar_points", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::PointCloud2::ConstSharedPtr message) {
        auto points = pointsFromCloud(*message);
        QMetaObject::invokeMethod(this, [this, points = std::move(points)]() mutable {
          showRadarData("livox", std::move(points));
        }, Qt::QueuedConnection);
      });
    scan_subscription_ = node_->create_subscription<sensor_msgs::msg::LaserScan>(
      "/scan", rclcpp::SensorDataQoS(),
      [this](sensor_msgs::msg::LaserScan::ConstSharedPtr message) {
        auto points = pointsFromScan(*message);
        QMetaObject::invokeMethod(this, [this, points = std::move(points)]() mutable {
          showRadarData("scan", std::move(points));
        }, Qt::QueuedConnection);
    });
  }
}

void SimulationControlSensorsPanel::showCameraFrame(const std::string & key, const QImage & image)
{
  auto found = camera_image_labels_.find(key);
  if (found == camera_image_labels_.end()) {
    if (camera_placeholder_ != nullptr) {
      camera_placeholder_->hide();
    }
    if (camera_tabs_ == nullptr) {
      camera_tabs_ = new QTabWidget(camera_page_);
      camera_tabs_->setObjectName(QStringLiteral("SimulationCameraTopics"));
      camera_page_->layout()->addWidget(camera_tabs_);
    }
    auto * image_widget = new CameraImageLabel(camera_tabs_);
    camera_tabs_->addTab(image_widget, QString::fromStdString(camera_labels_[key]));
    found = camera_image_labels_.emplace(key, image_widget).first;
  }
  auto * image_label = static_cast<CameraImageLabel *>(found->second);
  image_label->setImage(image);
}

void SimulationControlSensorsPanel::showRadarData(
  const std::string & key, std::vector<QPointF> points)
{
  auto found = radar_views_.find(key);
  if (found == radar_views_.end()) {
    auto * radar_view = new RadarView(radar_tabs_);
    const auto label = key == "livox" ? QStringLiteral("Livox 点云") : QStringLiteral("激光扫描 /scan");
    radar_tabs_->addTab(radar_view, label);
    found = radar_views_.emplace(key, radar_view).first;
  }
  radar_placeholder_->hide();
  radar_tabs_->show();
  found->second->setPoints(std::move(points));
}

void SimulationControlSensorsPanel::setRadarStatus(const QString & status)
{
  if (radar_placeholder_ != nullptr) {
    radar_placeholder_->setText(status);
  }
}

void SimulationControlSensorsPanel::save(rviz_common::Config config) const
{
  rviz_common::Panel::save(config);
  config.mapSetValue("Show Cameras", show_cameras_);
  config.mapSetValue("Show Lidar", show_lidar_);
  base_panel_->save(config.mapMakeChild(QStringLiteral("BasePanel")));
  lift_panel_->save(config.mapMakeChild(QStringLiteral("LiftPanel")));
}

void SimulationControlSensorsPanel::load(const rviz_common::Config & config)
{
  rviz_common::Panel::load(config);
  config.mapGetBool(QStringLiteral("Show Cameras"), &show_cameras_);
  config.mapGetBool(QStringLiteral("Show Lidar"), &show_lidar_);
  const auto base_config = config.mapGetChild(QStringLiteral("BasePanel"));
  if (base_config.isValid()) {
    base_panel_->load(base_config);
  }
  const auto lift_config = config.mapGetChild(QStringLiteral("LiftPanel"));
  if (lift_config.isValid()) {
    lift_panel_->load(lift_config);
  }
  if (rviz_initialized_) {
    buildSensorPage();
    subscribeToSelectedSensors();
  }
}

}  // namespace openflex_isaac_bringup

#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(
  openflex_isaac_bringup::SimulationControlSensorsPanel, rviz_common::Panel)
