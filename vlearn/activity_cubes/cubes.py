import os
import cv2
import math
import numpy as np
import pandas as pd
import pdb
from ..fd_ops import list_files
from ..vid_reader import VReader
import matplotlib.pyplot as plt


class ActCubes(VReader):
    def __init__(self, rdir, gt_csv_name):

        """
        Description:
            Initializes a pandas data frame with activity cubes. These are
            read from csv files present under root directory 
            (including sub-directories)
        Args:
            rdir (str): 
                Root directory containing csv files which contain activity cubes
            csv_name (str):
                CSV file name.
        Example:
            ```
            import vlearn
            ac = vlearn.ActCubes("./training")
            ```
        """
        self.rdir = rdir
        """ Root directory having ground truth csv file and videos """

        self.cubes = pd.DataFrame()
        """ Data frame having activity cube information """

        all_csv_files = list_files(rdir, [gt_csv_name])
        for idx, ccsv in enumerate(all_csv_files):
            if idx == 0:
                self.cubes = pd.read_csv(ccsv)
            else:
                tmp_cubes = pd.read_csv(ccsv)
                tmp_cubes = [tmp_cubes, self.cubes]
                self.cubes = pd.concat(tmp_cubes)

    def plot_properties(self):
        """
        Description:
            Creates histogram of  width, height and number of frames.
            These histograms are returned as list of axis handles. To
            view them please use `plt.imshow()`
        """
        warr = np.array(self.cubes["w"])
        harr = np.array(self.cubes["h"])
        farr = np.array(self.cubes["f"])

        wax = self.__plot_histogram(
            warr, "Width(pixels)", "Count", "Width histogram of activity cubes"
        )
        hax = self.__plot_histogram(
            harr, "Height(pixels)", "Count", "Height histogram of activity cubes"
        )
        fax = self.__plot_histogram(
            farr, "Number of frames", "Count", "No. frames histogram of activity cubes"
        )
        tax = self.__plot_histogram(
            farr / 30, "Time in seconds", "Count", "Play back time  of activity cubes"
        )
        plt.show()
        pdb.set_trace()

    def __plot_histogram(self, arr, xlab, ylab, title):
        """
        Description:
            Creates histogram using a numpy array.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        kwargs = dict(histtype="stepfilled", alpha=0.8, bins=20)
        ax.hist(arr, **kwargs)
        ax.set_xlabel(xlab, fontsize=30)
        ax.set_ylabel(ylab, fontsize=30)
        ax.tick_params(axis="both", which="major", labelsize=25)
        ax.tick_params(axis="both", which="minor", labelsize=25)
        ax.set_title(title, fontsize=35)
        return ax

    def extract_activity_cubes(self, frames=30, out_dir=""):
        """
        Trims activity cubes and stores them as videos. For
        writing and nowriting wenjing suggested to use 30
        frames or 1 second.
        Args:
            frames (int): Number of frames in each cube.
            out_dir (str): Output location of trimmed videos. If not passed videos
                are generated at the same location as video (csv file).
            np_array (boolen): When true generates a numpy array for all the trimmed
                videos.
        """
        for idx, row in self.cubes.iterrows():
            # Read video for current activity cube
            print(row)
            matlab_gt_name = row["name"]
            vid_name = matlab_gt_name.replace("_vj_gTruth", "")  # ??? Hard coded
            vpath = self.rdir + "/" + vid_name + ".mp4"
            super(ActCubes, self).__init__(vpath)
            # Trim videos every n frames
            ntrims = math.floor(row["f"] / frames)
            for ctrim in range(0, ntrims):
                poc = row["f0"] + frames * ctrim
                poce = poc + frames
                trim_name = vid_name + "_" + str(poc) + "_" + str(poce - 1)
                trim_path = self.rdir + "/" + vid_name + "_trimmed"
                if out_dir == "":
                    if not (os.path.isdir(trim_path)):
                        os.mkdir(trim_path)
                    if not (os.path.isdir(trim_path + "/" + row["activity"])):
                        os.mkdir(trim_path + "/" + row["activity"])
                    trim_fpath = (
                        trim_path
                        + "/"
                        + row["activity"]
                        + "/"
                        + row["person"]
                        + "_"
                        + trim_name
                        + ".avi"
                    )
                else:
                    if not (os.path.isdir(out_dir + "/trimmed_" + row["activity"])):
                        os.mkdir(out_dir + "/trimmed_" + row["activity"])
                    trim_fpath = (
                        out_dir
                        + "/trimmed_"
                        + row["activity"]
                        + "/"
                        + row["person"]
                        + "_"
                        + trim_name
                        + ".avi"
                    )
                vwr = cv2.VideoWriter(
                    trim_fpath,
                    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                    row["FPS"],
                    (100, 100),
                )
                self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)
                while self.vro.isOpened() and poc < poce:
                    ret, frm = self.vro.read()
                    trm_frm = frm[
                        int(row["h0"]) : int(row["h0"] + row["h"]),
                        int(row["w0"]) : int(row["w0"] + row["w"]),
                    ]
                    trm_frm = cv2.resize(trm_frm, (100, 100))
                    vwr.write(trm_frm)
                    poc = poc + 1
                vwr.release()
            self.vro.release()

    def extract_activity_cubes_as_np_arrays(self, frames=30, out_dir=""):
        """
        Trims activity cubes and stores them as videos. For
        writing and nowriting wenjing suggested to use 30
        frames or 1 second.
        Args:
            frames (int): Number of frames in each cube.
            out_dir (str): Output location of trimmed videos. If not passed videos
                are generated at the same location as video (csv file).
            np_array (boolen): When true generates a numpy array for all the trimmed
                videos.
        """
        num_gt_cubes = len(self.cubes)
        gray_list = []
        for idx, row in self.cubes.iterrows():
            # Read video for current activity cube
            print(row)
            matlab_gt_name = row["name"]
            vid_name = matlab_gt_name.replace("_vj_gTruth", "")  # ??? Hard coded
            vpath = self.rdir + "/" + vid_name + ".mp4"
            super(ActCubes, self).__init__(vpath)
            # Trim videos every n frames
            ntrims = math.floor(row["f"] / frames)
            for ctrim in range(0, ntrims):
                poc = row["f0"] + frames * ctrim
                poce = poc + frames
                trim_name = vid_name + "_" + str(poc) + "_" + str(poce - 1)
                trim_path = self.rdir + "/" + vid_name + "_trimmed"
                if out_dir == "":
                    if not (os.path.isdir(trim_path)):
                        os.mkdir(trim_path)
                    if not (os.path.isdir(trim_path + "/" + row["activity"])):
                        os.mkdir(trim_path + "/" + row["activity"])
                    trim_fpath = (
                        trim_path
                        + "/"
                        + row["activity"]
                        + "/"
                        + row["person"]
                        + "_"
                        + trim_name
                        + ".npy"
                    )
                else:
                    if not (os.path.isdir(out_dir + "/trimmed_" + row["activity"])):
                        os.mkdir(out_dir + "/trimmed_" + row["activity"])
                    trim_fpath = (
                        out_dir
                        + "/trimmed_"
                        + row["activity"]
                        + "/"
                        + row["person"]
                        + "_"
                        + trim_name
                        + ".npy"
                    )
                self.vro.set(cv2.CAP_PROP_POS_FRAMES, poc)
                k = 0
                trm_arr = np.ndarray((100, 100, 3, frames))
                while self.vro.isOpened() and poc < poce:
                    ret, frm = self.vro.read()
                    trm_frm = frm[
                        int(row["h0"]) : int(row["h0"] + row["h"]),
                        int(row["w0"]) : int(row["w0"] + row["w"]),
                    ]
                    trm_frm = cv2.resize(trm_frm, (100, 100))
                    trm_arr[:, :, :, k] = trm_frm
                    poc = poc + 1
                    k = k + 1
                np.save(trim_fpath, trm_arr)
            self.vro.release()
