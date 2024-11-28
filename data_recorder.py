import glob
import os
import sys

import carla.libcarla

try:
    sys.path.append(glob.glob('/home/abdou/CARLA_0.9.15/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import logging

import random
import time
from pathlib import Path

import queue
from queue import Empty

import imageio.v2 as imageio
import numpy as np

from flow_vis import flow_to_color, flow_uv_to_colors
import cv2
import open3d as o3d
from matplotlib import cm
from matplotlib import colors

from functools import partial
import weakref

def visualize_optical_flow(flow, return_image=False, text=None, scaling=None):
    # flow -> numpy array 2 x height x width
    # 2,h,w -> h,w,2
    # flow = flow.transpose(1,2,0)
    flow[np.isinf(flow)]=0
    # Use Hue, Saturation, Value colour model
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=float)

    # The additional **0.5 is a scaling factor
    mag = np.sqrt(flow[...,0]**2+flow[...,1]**2)**0.5

    ang = np.arctan2(flow[...,1], flow[...,0])
    ang[ang<0]+=np.pi*2
    hsv[..., 0] = ang/np.pi/2.0 # Scale from 0..1
    hsv[..., 1] = 1
    if scaling is None:
        hsv[..., 2] = (mag-mag.min())/(mag-mag.min()).max() # Scale from 0..1
    else:
        mag[mag>scaling]=scaling
        hsv[...,2] = mag/scaling
    rgb = colors.hsv_to_rgb(hsv)

    return rgb, (mag.min(), mag.max())


# @todo cannot import these directly.
SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []

class DataCollector():
    # The data collector must connect with CARLA, retreive the world and pass everything to each DataRecorder
    # A separate data recorder is used for each Scenario to record a sequence
    def __init__(self, global_config, sensors_config, scenarios):
        # config
        self.config = global_config
        self.sensors_config = sensors_config
        self.scenarios = scenarios

        # save_dir
        self.save_dir = Path(self.config["collector"]["save_dir"])
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # client
        # self.client =  carla.Client(self.config["client"]["host"], self.config["client"]["port"])
        # self.client.set_timeout(self.config["client"]["timeout"])


    def collect(self):
        # Get maps
        for scenario_args in self.scenarios.values():
            recorder = DataRecorder(scenario_args, self.config, self.sensors_config)
            recorder.record()

def record_scenario(scenario_id:int, global_config, sensors_config, scenarios):
    scenario_name = f"Scenario {scenario_id}"
    assert scenario_name in scenarios

    recorder = DataRecorder(scenarios[scenario_name], global_config, sensors_config)
    recorder.record()
    time.sleep(1)

class DataRecorder():
    def __init__(self, args, global_config, sensors_config):     
        self.args = args
        self.global_config = global_config
        self.sensors_config = sensors_config

        rand_seed = self.global_config["collector"]["random_seed"]

        save_dir_root = global_config["collector"]["save_dir"]
        self.save_dir = Path(save_dir_root) / args["name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.duration = args["recording_duration"] * 60

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        # Client and world
        host = global_config["client"]["host"]
        port = global_config["client"]["port"]
        self.client = carla.Client(host, port)
        self.client.set_timeout(global_config["client"]["timeout"])
        
        # world and traffic manager
        self.world = self.client.load_world(args['map'])

        # spectator
        self.spectator = self.world.get_spectator()

        self.traffic_manager = self.client.get_trafficmanager(global_config["traffic_manager"]["tm_port"])
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        if global_config["traffic_manager"]["respawn"]:
            self.traffic_manager.set_respawn_dormant_vehicles(True)
        if global_config["traffic_manager"]["hybrid"]:
            self.traffic_manager.set_hybrid_physics_mode(True)
            self.traffic_manager.set_hybrid_physics_radius(70.0)
        if rand_seed is not None:
            self.traffic_manager.set_random_device_seed(rand_seed)
        
        self.synchronous_master = False
        random.seed(rand_seed if rand_seed is not None else int(time.time()))

        # Activate synchronous mode with a fixed time step
        settings = self.world.get_settings()
        if global_config["simulation"]["synchronous"]:
            self.synchronous_master = True
            self.traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = global_config["simulation"]["time_step"]

        # Rendering
        if global_config["simulation"]["no_rendering"]:
            settings.no_rendering_mode = True
        self.world.apply_settings(settings)

        # Weather settings
        weather = carla.WeatherParameters(**args["weather"])
        self.world.set_weather(weather)

        # Blueprints
        self.blueprints = get_actor_blueprints(self.world, global_config["simulation"]["filterv"], global_config["simulation"]["generationv"])
        self.blueprintsWalkers = get_actor_blueprints(self.world, global_config["simulation"]["filterw"], global_config["simulation"]["generationw"])
        
        if global_config["traffic_manager"]["safe"]:
            self.blueprints = [x for x in self.blueprints if x.get_attribute('base_type') == 'car']

        self.blueprints = sorted(self.blueprints, key=lambda bp: bp.id)

        # Spawn Points
        self.spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.spawn_points)

        if args["number_of_vehicles"] < number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif args["number_of_vehicles"] > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args["number_of_vehicles"], number_of_spawn_points)
            args["number_of_vehicles"] = number_of_spawn_points

        # Spwan vehicles And Walkers

        # Start with the main car to which the sensors are to be attached
        hero_bp = self.world.get_blueprint_library().find('vehicle.audi.tt')
        hero_transform = random.choice(self.spawn_points)
        self.hero = self.world.spawn_actor(hero_bp, hero_transform)
        self.hero.set_autopilot(True)
        self.traffic_manager.auto_lane_change(self.hero, False)

        # Spawn npc vehicles and walkers
        self.spawn_vehicles()
        self.spawn_walkers()

        # Example of how to use Traffic Manager parameters
        # self.traffic_manager.global_percentage_speed_difference(30.0)

        # Sensors
        # available_sensors = []
        # sensors_seetings = global_config["sensors"]
        # enabled_sensors = sensors_seetings["enable"]
        # self.visualize = sensors_seetings["visualize"]

        # self.visualize = {'optical_flow':True,
        #                   'semantic_segmentation':True,
        #                   'dvs':True}
        
        # self.lidar = 'lidar' in enabled_sensors             
        # self.sensors = {key : None for key in enabled_sensors}
        # self.sensors_bp = {key : None for key in enabled_sensors}
        # self.sensor_queues = {key:queue.Queue() for key in enabled_sensors}

        # self.sensor_names = enabled_sensors
        self.sensors = {}
        self.sensors_bp = {}
        self.data_save_dirs = {}
        self.visual_dirs = {}
        self.sensor_queues = {}
        self.spawn_sensors()

        self.first_frame = None
        self.rgb_timestamps = []

        # Events cumulator (under test)
        self.events_cumulator = {'t' : [], 'x' : [], 'y' : [], 'pol' : []}

        self.rgb_count = 0
        self.start_time = None
        return
        
    def spawn_sensors(self):

        # if self.lidar:
        #     lidar_settings = self.global_config["sensors"]["lidar"]
        sensors_location = carla.Location(**self.global_config["sensors"]["location"])

        for s_name, conf in self.sensors_config.items():
            if not conf['enable']:
                continue
            sensor_bp = self.world.get_blueprint_library().find(conf['blueprint_name'])
            for key, value in conf['settings'].items():
                sensor_bp.set_attribute(key, str(value))

            camera_transform = carla.Transform(sensors_location)
            sensor = self.world.spawn_actor(sensor_bp, camera_transform, attach_to = self.hero,
                                            attachment_type = carla.libcarla.AttachmentType.Rigid)
            
            self.sensors[s_name] = sensor
            self.sensors_bp[s_name] = sensor_bp

            self.data_save_dirs[s_name] = self.save_dir / s_name

            if conf['visualize']:
                self.visual_dirs[s_name] = self.save_dir / 'visualizations' / s_name
            else:
                self.visual_dirs[s_name] = None

            self.sensor_queues[s_name] = queue.Queue()

            print(f'Created {s_name} of type : {sensor.type_id}')


        for dir in self.data_save_dirs.values():
            if dir is not None:
                dir.mkdir(parents=True, exist_ok=True)

        for dir in self.visual_dirs.values():
            if dir is not None:
                dir.mkdir(parents=True, exist_ok=True)
        
        # for s_name in self.sensor_names:
        #     if  s_name == "lidar" and self.lidar:
        #         # Create Lidar
        #         lidar_bp = self.world.get_blueprint_library().find("sensor.lidar.ray_cast")
        #         for key, value in lidar_settings.items():
        #             lidar_bp.set_attribute(key, str(value))

        #         lidar_transform = carla.Transform(sensors_location)
        #         self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to = self.hero,
        #                                             attachment_type = carla.libcarla.AttachmentType.Rigid)
        #         self.sensors[s_name] = self.lidar
        #         self.sensors_bp[s_name] = self.lidar_bp

        #         self.lidar_point_list = o3d.geometry.PointCloud()
        #         print(f'Created {self.lidar.type_id}')

    
    def spawn_vehicles(self):
        batch = []
        hero = False # Hero is spawned separately
        for n, transform in enumerate(self.spawn_points):
            if n >= self.args["number_of_vehicles"]:
                break
            blueprint = random.choice(self.blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
                        
        for response in self.client.apply_batch_sync(batch, self.synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if self.args["car_lights_on"]:
            all_vehicle_actors = self.world.get_actors(self.vehicles_list)
            for actor in all_vehicle_actors:
                self.traffic_manager.update_vehicle_lights(actor, True)


        # Disabling auto change lane (didn't help)
        vehicles = self.world.get_actors(self.vehicles_list)
        for vehicle in vehicles:
            self.traffic_manager.auto_lane_change(vehicle, False)


    def spawn_walkers(self):
        # some settings
        percentagePedestriansRunning = 0.0      # how many pedestrians will run
        percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
        if self.args["seed_walkers"]:
            self.world.set_pedestrians_seed(self.args["seed_walkers"])
            random.seed(self.args["seed_walkers"])
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(self.args["number_of_walkers"]):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(self.blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(self.walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), self.walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                self.walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(self.walkers_list)):
            self.all_id.append(self.walkers_list[i]["con"])
            self.all_id.append(self.walkers_list[i]["id"])


        self.all_actors = self.world.get_actors(self.all_id)

        # wait for a tick to ensure client receives the last transform of the walkers we have just created
        # if False:
        #     self.world.wait_for_tick()
        # else:
        #     self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(self.all_id), 2):
            # start walker
            self.all_actors[i].start()
            # set walk to random point
            self.all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            self.all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))


    def start_recording(self):
        for s_name, sensor in self.sensors.items():
            # Start all sensors
            sensor.listen(self.get_callback(s_name))

    def stop_recording(self):
        # Stop all sensors
        for sensor in self.sensors.values():
            sensor.stop()

    def get_relative_frame(self, frame):
        if self.first_frame is None:
            return 0
        
        return frame - self.first_frame

    def get_callback(self, s_name):
        weak_self = weakref.ref(self)

        if s_name == 'RGB':
            return partial(self.rgb_callback, weak_self = weak_self, sensor = "RGB")
        
        if s_name == 'RGB_1000':
            return partial(self.rgb_1000_callback, weak_self = weak_self, sensor = "RGB_1000")

        elif s_name == 'SEMANTIC_SEGMENTATION':
            return partial(self.segmentation_callback, weak_self = weak_self, sensor = "SEMANTIC_SEGMENTATION")

        elif s_name == "OPTICAL_FLOW":
            return partial(self.flow_callback, weak_self = weak_self, sensor = "OPTICAL_FLOW")
        
        elif s_name == "DVS":
            return partial(self.dvs_callback, weak_self = weak_self, sensor = "DVS")
        
        elif s_name == "LIDAR" :
            return partial(self.lidar_callback, weak_self = weak_self, sensor = "LIDAR")
        else:
            raise NotImplementedError
        
    
    @staticmethod
    def rgb_callback(image, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]
        
        frame = self.get_relative_frame(image.frame)
        frame_file_name = '{:06d}.png'.format(frame)
        image.save_to_disk(str(save_dir / frame_file_name))

        sensor_queue.put((frame, 'rgb_camera'))

    @staticmethod
    def rgb_1000_callback(image, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]
        
        frame = self.get_relative_frame(image.frame)
        frame_file_name = '{:06d}.png'.format(frame)
        image.save_to_disk(str(save_dir / frame_file_name))

        if self.rgb_count == 0:
            self.start_time = time.time()

        self.rgb_count += 1

        if self.rgb_count == 1000:
            print(f"It took {time.time() - self.start_time} seconds to record 1000 frames")
            self.rgb_count = 0

        sensor_queue.put((frame, 'rgb_camera_1000'))


    @staticmethod
    def segmentation_callback(segmentation, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]

        frame = self.get_relative_frame(segmentation.frame)
        frame_file_name = '{:06d}.png'.format(frame)
        segmentation.save_to_disk(str(save_dir / frame_file_name))

        vis_dir = self.visual_dirs[sensor]
        if vis_dir is not None:
            segmentation.save_to_disk(str(vis_dir / frame_file_name), carla.ColorConverter.CityScapesPalette)

        sensor_queue.put((frame, 'segmentation_camera'))


    @staticmethod
    def flow_callback(flow, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]

        frame = self.get_relative_frame(flow.frame)
        frame_file_name = '{:06d}.png'.format(frame)

        raw = np.frombuffer(flow.raw_data, dtype=np.float32)

        # print(np.min(raw), np.max(raw), np.mean(raw))

        raw = raw.reshape((flow.height, flow.width, 2))

        # Flow values are in the range [-2,2] so it must be scaled
        # we multiply the y component by -1 to get the forward flow (carla documentation)
        flow_uv = np.ndarray((flow.height, flow.width, 3))
        flow_uv[:,:,0] = raw[:,:,0] * 0.5 * flow.width
        flow_uv[:,:,1] = raw[:,:,1] * -0.5 * flow.height

        # Visualize
        vis_dir = self.visual_dirs[sensor]
        if vis_dir is not None:
            # rgb, _ = visualize_optical_flow(flow_uv[:,:,:2])
            # rgb *= 255
            # imageio.imwrite(str(vis_dir / f'vis_{flow.frame}.png'), rgb.astype('uint8'))
            vis = flow_uv_to_colors(u = flow_uv[:,:,0], v = flow_uv[:,:,1])
            imageio.imwrite(str(vis_dir / frame_file_name), vis.astype('uint8'))

        # Save flow
        flow_uv = flow_uv * 128.0 + 2**15 
        flow_uv[:,:,2] = 1
        imageio.imwrite(str(save_dir / frame_file_name), flow_uv.astype(np.uint16), format='PNG-FI')

        sensor_queue.put((frame, 'optical_flow_camera'))


    @staticmethod
    def dvs_callback(events, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]

        frame = self.get_relative_frame(events.frame)
        events_file_name = '{:06d}.npy'.format(frame)

        dvs_events = np.frombuffer(events.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
        
        # Cumulate events
        for key in self.events_cumulator:
            self.events_cumulator[key].append(dvs_events[:][key].copy())
        
        if len(self.events_cumulator['x']) >= 100:
            x = np.concatenate(self.events_cumulator['x'])
            y = np.concatenate(self.events_cumulator['y'])
            t = np.concatenate(self.events_cumulator['t'])
            p = np.concatenate(self.events_cumulator['pol']).astype(int)

            if sensor in self.visualize:
                vis_dir = self.visual_dirs[sensor]
                dvs_image = np.zeros((events.height, events.width, 3), dtype=np.uint8)
                dvs_image[y[:], x[:], p[:] * 2] = 255
                imageio.imwrite(str(vis_dir / '{:06d}.png'.format(frame)), dvs_image)

            for key in self.events_cumulator:
                self.events_cumulator[key] = []

        sensor_queue.put((frame, 'dvs_camera'))

        return
        
        # Save events
        x = dvs_events[:]['x']
        y = dvs_events[:]['y']
        t = dvs_events[:]['t']
        p = dvs_events[:]['pol'].astype(int)

        events_stream = np.stack([x, y, t, p], axis=1)
        np.save(str(save_dir / events_file_name), events_stream)

        
        # Visualize events
        if sensor in self.visualize:
            vis_dir = self.visual_dirs[sensor]
            dvs_image = np.zeros((events.height, events.width, 3), dtype=np.uint8)
            dvs_image[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            imageio.imwrite(str(vis_dir / '{:06d}.png'.format(frame)), dvs_image)

        
        # TODO : Append trajectory point (location, velocity, acceleration)
        #        and save the whole trajectory at the end of the recording



    @staticmethod
    def lidar_callback(point_cloud, weak_self, sensor):
        self = weak_self()
        save_dir = self.data_save_dirs[sensor]
        sensor_queue = self.sensor_queues[sensor]
        point_list = self.lidar_point_list

        frame = self.get_relative_frame(point_cloud.frame)

        # Save Point Cloud file
        point_cloud.save_to_disk(str(save_dir / '{:06d}.ply'.format(frame)))

        sensor_queue.put((frame, 'lidar'))
        return
        
        if sensor in self.visualize:
            vis_dir = self.visual_dirs[sensor]

            data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
            data = np.reshape(data, (int(data.shape[0] / 4), 4)) # x, y, z, dist_from_sensor

            # Isolate the intensity and compute color for it
            intensity = data[:, -1]
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
            int_color = np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
            
            points = data[:, :-1]
            points[:, :1] = -points[:, :1]

            point_list.points = o3d.utility.Vector3dVector(points)
            point_list.colors = o3d.utility.Vector3dVector(int_color)


    def record(self):
        try:
            print(f"Recording {self.args['name']} [Duration : {self.duration} seconds] ... press ctl+c to force exit")
            end_time = time.time() + self.duration
            self.start_recording()
            first = True
            while True:
                if self.global_config["simulation"]["synchronous"]:
                    self.world.tick()
                    w_frame = self.world.get_snapshot().frame
                    # print("\nWorld's frame: {}".format(w_frame))

                    if first:
                        # Save the first frame so the numbering can start from 0
                        self.first_frame = w_frame
                        first = False

                    # Wait for all data to be written to disk.
                    try:
                        s_frame = self.sensor_queues['RGB_1000'].get(True, 1.0)

                        # for queue in self.sensor_queues.values():
                        #     s_frame = queue.get(True, 1.0)
                    except Empty:
                        print("Some of the sensor information is missed")
                else:
                    self.world.wait_for_tick()

                if time.time() > end_time:
                    break
            self.stop_recording()

        except RuntimeError as err:
            if "time-out" in str(err):
                print(f"Time-out error encountred while recording {self.args['name']}")

        except KeyboardInterrupt:
            self.stop_recording()
            print("Quitting...")
            pass

        except:
            pass

        finally:
            # Disable Synchronous mode and reactivate rendering
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

            # Destroy vehicles
            print('\ndestroying %d sensors' % len(self.sensors))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.sensors.values()])

            # Destroy vehicles
            print('\ndestroying %d vehicles' % len(self.vehicles_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])

            # stop walker controllers (list is [controller, actor, controller, actor ...])
            for i in range(0, len(self.all_id), 2):
                self.all_actors[i].stop()

            # Destroy all walkers
            print('\ndestroying %d walkers' % len(self.walkers_list))
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.all_id])

            time.sleep(3)