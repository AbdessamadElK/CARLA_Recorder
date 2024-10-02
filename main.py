import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import argparse
import logging

import yaml

from data_recorder import DataCollector

GLOBAL_CONFIG = "./config/global.yaml"
SENSORS_CONFIG = "./config/sensors.yaml"
SCENARIOS_CONFIG = "./config/scenarios.yaml"

if __name__ == "__main__":

    with open(GLOBAL_CONFIG) as global_cfg:
        global_config = yaml.safe_load(global_cfg)

    with open(SCENARIOS_CONFIG) as scenarios_cfg:
        scenarios = yaml.safe_load(scenarios_cfg)

    collector = DataCollector(global_config, scenarios)
    collector.collect()
