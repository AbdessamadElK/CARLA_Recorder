import carla
import numpy as np

import imageio.v2 as imageio
from flow_vis import flow_uv_to_colors

import time

def dummy_callback(data, weak_self, sensor):
    """
        Just a dummy callback to be loaded when the actual callback is not defined.
        This allows the simulation to run, but the sensor for which the callback is
        not defined will not work.
    """
    return

def rgb_callback(image, weak_self, sensor):
    recorder = weak_self()
    save_dir = recorder.data_save_dirs[sensor]
    sensor_queue = recorder.sensor_queues[sensor]
    
    frame = recorder.get_relative_frame(image.frame)
    frame_file_name = '{:06d}.png'.format(frame)
    image.save_to_disk(str(save_dir / frame_file_name))

    sensor_queue.put((frame, 'rgb_camera'))

def rgb_1000_callback(image, weak_self, sensor):
    recorder = weak_self()
    save_dir = recorder.data_save_dirs[sensor]
    sensor_queue = recorder.sensor_queues[sensor]
    
    frame = recorder.get_relative_frame(image.frame)
    frame_file_name = '{:06d}.png'.format(frame)
    image.save_to_disk(str(save_dir / frame_file_name))

    if recorder.rgb_count == 0:
        recorder.start_time = time.time()

    recorder.rgb_count += 1

    if recorder.rgb_count == 100:
        rgb_10_save_dir = save_dir.parent / 'RGB_10'
        rgb_10_save_dir.mkdir(parents=True, exist_ok=True)

        image.save_to_disk(str(rgb_10_save_dir/ frame_file_name))
        recorder.rgb_count = 0
        # print(f"It took {time.time() - recorder.start_time} seconds to record 100 frames")

    sensor_queue.put((frame, 'rgb_camera_1000'))


def semantic_callback(segmentation, weak_self, sensor):
    recorder = weak_self()
    save_dir = recorder.data_save_dirs[sensor]
    sensor_queue = recorder.sensor_queues[sensor]

    frame = recorder.get_relative_frame(segmentation.frame)
    frame_file_name = '{:06d}.png'.format(frame)
    segmentation.save_to_disk(str(save_dir / frame_file_name))

    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        segmentation.save_to_disk(str(vis_dir / frame_file_name), carla.ColorConverter.CityScapesPalette)

    sensor_queue.put((frame, 'segmentation_camera'))


def flow_callback(flow, weak_self, sensor):
    recorder = weak_self()
    save_dir = recorder.data_save_dirs[sensor]
    sensor_queue = recorder.sensor_queues[sensor]

    frame = recorder.get_relative_frame(flow.frame)
    frame_file_name = '{:06d}.png'.format(frame)


    raw = np.array([(pixel.x, pixel.y) for pixel in flow], dtype=np.float64)
    raw = raw.reshape((flow.height, flow.width, 2))
    # raw = np.frombuffer(flow.raw_data, dtype=np.float32)

    # print(np.min(raw), np.max(raw), np.mean(raw))

    # raw = raw.reshape((flow.height, flow.width, 2))

    # Flow values are in the range [-2,2] so it must be scaled
    # we multiply the y component by -1 to get the forward flow (carla documentation)
    flow_uv = np.ndarray((flow.height, flow.width, 3))
    
    # flow_uv[:,:,0] = raw[:,:,0]
    # flow_uv[:,:,1] = raw[:,:,1] * -1.0

    flow_uv[:,:,0] = raw[:,:,0] * 0.5 * float(flow.width)
    flow_uv[:,:,1] = raw[:,:,1] * 0.5 * float(flow.height) * -1
    
    # Visualize
    vis_dir = recorder.visual_dirs[sensor]
    if vis_dir is not None:
        # rgb, _ = visualize_optical_flow(flow_uv[:,:,:2])
        # rgb *= 255
        # imageio.imwrite(str(vis_dir / f'vis_{flow.frame}.png'), rgb.astype('uint8'))
        BUILTIN  = False
        if BUILTIN:
            image = flow.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            imageio.imwrite(str(vis_dir / frame_file_name), array)
        else:
            vis = flow_uv_to_colors(u = flow_uv[:,:,0], v = flow_uv[:,:,1])
            imageio.imwrite(str(vis_dir / frame_file_name), vis.astype('uint8'))



    # Save flow
    flow_uv = flow_uv * 128.0 + 2**15 
    flow_uv[:,:,2] = 1
    imageio.imwrite(str(save_dir / frame_file_name), flow_uv.astype(np.uint16), format='PNG-FI')

    sensor_queue.put((frame, 'optical_flow_camera'))


def dvs_callback(events, weak_self, sensor):
        recorder = weak_self()
        save_dir = recorder.data_save_dirs[sensor]
        sensor_queue = recorder.sensor_queues[sensor]

        frame = recorder.get_relative_frame(events.frame)
        events_file_name = '{:06d}.npz'.format(frame)

        dvs_events = np.frombuffer(events.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
        
        dvs_events_output = {key : dvs_events[:][key] for key in ['x', 'y', 't', 'pol']}

        np.savez(save_dir / events_file_name, **dvs_events_output)

        
        # Cumulate events
        # for key in recorder.events_cumulator:
        #     recorder.events_cumulator[key].append(dvs_events[:][key].copy())
        
        # if len(recorder.events_cumulator['x']) >= 100:
        #     x = np.concatenate(recorder.events_cumulator['x'])
        #     y = np.concatenate(recorder.events_cumulator['y'])
        #     t = np.concatenate(recorder.events_cumulator['t'])
        #     p = np.concatenate(recorder.events_cumulator['pol']).astype(int)

        #     if sensor in recorder.visualize:
        #         vis_dir = recorder.visual_dirs[sensor]
        #         dvs_image = np.zeros((events.height, events.width, 3), dtype=np.uint8)
        #         dvs_image[y[:], x[:], p[:] * 2] = 255
        #         imageio.imwrite(str(vis_dir / '{:06d}.png'.format(frame)), dvs_image)

            # for key in recorder.events_cumulator:
            #     recorder.events_cumulator[key] = []

        sensor_queue.put((frame, 'dvs_camera'))
