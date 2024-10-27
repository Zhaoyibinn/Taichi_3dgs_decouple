import rerun as rr  # pip install rerun-sdk
import open3d as o3d
import numpy as np
rr.init("rerun_example_app")

rr.connect()  # Connect to a remote viewer
# rr.spawn()  # Spawn a child process with a viewer and connect
# rr.save("recording.rrd")  # Stream all logs to disk
cloud = o3d.io.read_point_cloud("result/B330/ours_filtered.ply")
# Associate subsequent data with 42 on the “frame” timeline
positions = np.array(cloud.points)
for frame_idx in range(1000):
    rr.set_time_sequence("frame", frame_idx)
    beishu = 1 + frame_idx / 1000
    # Log colored 3D points to the entity at `path/to/points`
    rr.log("path/to/points", rr.Points3D(positions * beishu, colors=[255,255,255]))