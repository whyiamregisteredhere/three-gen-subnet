import os
import glob
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import imageio
import tqdm

from DreamGaussianLib import HDF5Loader
from DreamGaussianLib.CameraUtils import OrbitCamera, orbit_camera
from DreamGaussianLib.GaussianSplattingRenderer import GSRenderer, BasicCamera


class VideoUtils:
    def __init__(
        self,
        img_width: int = 512,
        img_height: int = 512,
        cam_rad: float = 2,
        azim_step: int = 5,
        elev_step: int = 20,
        elev_start: int = -60,
        elev_stop: int = 30,
    ):
        self.__img_width = img_width
        self.__img_height = img_height
        self.__cam_rad = cam_rad
        self.__azim_step = azim_step
        self.__elev_step = elev_step
        self.__elev_start = elev_start
        self.__elev_stop = elev_stop

    def render_gaussian_splatting_video(self, data_dir: str, out_dir: str, video_fps: int = 24):
        """Render Gaussian splatting videos from point cloud data."""
        hdf5loader = HDF5Loader.HDF5Loader()
        files = glob.glob(f"{data_dir}/*_pcl.h5")

        os.makedirs(out_dir, exist_ok=True)

        for file_path in files:
            name = os.path.basename(file_path)
            video_name = name.replace(".h5", ".mp4")
            video_path = os.path.join(out_dir, video_name)
            substring = "_".join(name.split("_")[:-1])

            print(f"[INFO] Processing {name}...")
            data_dict = hdf5loader.load_point_cloud_from_h5(substring, data_dir)

            renderer = GSRenderer()
            renderer.initialize(data_dict)

            orbitcam = OrbitCamera(self.__img_width, self.__img_height, r=self.__cam_rad, fovy=49.1)

            # Use direct streaming to avoid holding large image lists in memory
            self._stream_video(renderer, orbitcam, video_path, video_fps)

    def _stream_video(self, renderer: GSRenderer, orbitcam: OrbitCamera, video_path: str, fps: int):
        """Stream frames directly to video to save memory."""
        with imageio.get_writer(video_path, fps=fps) as writer:
            with tqdm.tqdm(total=self._total_frames(), desc="Rendering frames") as pbar:
                for elev, azimd in self._generate_camera_angles():
                    frame = self._render_frame(renderer, orbitcam, elev, azimd)
                    writer.append_data(frame)
                    pbar.update(1)

    def _render_frame(self, renderer: GSRenderer, orbitcam: OrbitCamera, elev: int, azimd: int):
        """Render a single frame."""
        pose = orbit_camera(elev, azimd, self.__cam_rad)
        camera = BasicCamera(
            pose,
            self.__img_width,
            self.__img_height,
            orbitcam.fovy,
            orbitcam.fovx,
            orbitcam.near,
            orbitcam.far,
        )
        output_dict = renderer.render(camera)
        img = output_dict["image"].permute(1, 2, 0).detach().cpu().numpy() * 255
        return img.astype(np.uint8)

    def _generate_camera_angles(self):
        """Generate camera angles for rendering."""
        for elev in range(self.__elev_start, self.__elev_stop, self.__elev_step):
            for azimd in range(0, 360, self.__azim_step):
                yield elev, azimd

    def _total_frames(self):
        """Calculate the total number of frames to be rendered."""
        return (
            (self.__elev_stop - self.__elev_start) // self.__elev_step
        ) * (360 // self.__azim_step)

    def render_video(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        features_dc: np.ndarray,
        features_rest: np.ndarray,
        opacities: np.ndarray,
        scale: np.ndarray,
        rotation: np.ndarray,
        sh_degree: int,
    ) -> BytesIO:
        """Render video from given point cloud data and return it as a BytesIO object."""
        data_dict = {
            "points": points,
            "normals": normals,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "scale": scale,
            "rotation": rotation,
            "sh_degree": sh_degree,
        }
        renderer = GSRenderer()
        renderer.initialize(data_dict)

        orbitcam = OrbitCamera(self.__img_width, self.__img_height, r=self.__cam_rad, fovy=49.1)

        buffer = BytesIO()
        with imageio.get_writer(buffer, format="mp4", mode="I", fps=24) as writer:
            for elev, azimd in self._generate_camera_angles():
                frame = self._render_frame(renderer, orbitcam, elev, azimd)
                writer.append_data(frame)

        buffer.seek(0)
        return buffer
