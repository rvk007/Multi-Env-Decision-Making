import os
import logging
import imageio


class VideoRecorder:
    def __init__(self, root_dir, fps=5):
        logging.getLogger('imageio_ffmpeg').setLevel(logging.ERROR)
        self.save_dir = os.path.join(root_dir, 'eval_video') if root_dir else None
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
        self.fps = fps
        self.frames = []

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env):
        if self.enabled:
            frame = env.render(mode='rgb_array')
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.save_dir, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps, macro_block_size=10, ffmpeg_params=['-loglevel', 'error'])
