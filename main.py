import glob
import os
import sys

# try:
#     sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
# except IndexError:
#     pass

import argparse
import logging

import yaml

from data_recorder import DataCollector, record_scenario

GLOBAL_CONFIG = "./config/global.yaml"
SENSORS_CONFIG = "./config/sensors.yaml"
SCENARIOS_CONFIG = "./config/scenarios.yaml"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", "-s", type=int, default="", help="The desired scenario's id (see scenarios.yaml for defined scenarios)")
    args = parser.parse_args()

    with open(GLOBAL_CONFIG) as global_cfg:
        global_config = yaml.safe_load(global_cfg)

    with open(SENSORS_CONFIG) as sensors_cfg:
        sensors_config = yaml.safe_load(sensors_cfg)

    with open(SCENARIOS_CONFIG) as scenarios_cfg:
        scenarios = yaml.safe_load(scenarios_cfg)


    if args.scenario == "":
        collector = DataCollector(global_config, sensors_config, scenarios)
        collector.collect()
    else:
        record_scenario(args.scenario, global_config, sensors_config, scenarios)

