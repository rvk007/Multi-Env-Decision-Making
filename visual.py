from gym.wrappers import Monitor


def record_videos(env, path):
    monitor = Monitor(env, path, force=True, video_callable=lambda episode: True)
    env.unwrapped.set_monitor(monitor)  # Capture intermediate frames
    return monitor
