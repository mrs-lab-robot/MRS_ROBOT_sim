#!/usr/bin/env bash
set -eo pipefail

workspace="${OPENFLEX_RVIZ_WS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
domain_id="${1:-${ROS_DOMAIN_ID:-1}}"
config="${RVIZ_CONFIG:-${workspace}/install/openarmx_integrated_description/share/openarmx_integrated_description/rviz/integrated_robot.rviz}"

if [[ ! "${domain_id}" =~ ^[0-9]+$ ]] || ((domain_id > 232)); then
  echo "ROS_DOMAIN_ID must be an integer from 0 to 232: ${domain_id}" >&2
  exit 1
fi

if [[ ! -f "${workspace}/install/setup.bash" ]]; then
  echo "RViz workspace is not built: ${workspace}" >&2
  exit 1
fi
if [[ ! -f "${config}" ]]; then
  echo "RViz config does not exist: ${config}" >&2
  exit 1
fi

source /opt/ros/humble/setup.bash
source "${workspace}/install/setup.bash"
export ROS_DOMAIN_ID="${domain_id}"
export ROS_LOCALHOST_ONLY=0

echo "Starting OpenFleX RViz: workspace=${workspace} domain=${ROS_DOMAIN_ID}"
exec rviz2 -d "${config}"
