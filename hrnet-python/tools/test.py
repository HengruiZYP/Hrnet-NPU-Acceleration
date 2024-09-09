import os, sys

upper_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(upper_level_path)
from hrnet import KeyPointDetector, VisKeypointDetection
import yaml
import cv2
import paddle
import paddle.fluid as fluid
import types
import os
import argparse
import copy
import pickle
import numpy as np
import time
class PaddleInfer(KeyPointDetector):
    """Paddle Inference """
    def __init__(self, model_dir, config, infer_yml):
        """init

        Args:
            model_dir (str): model_dir
            config (str): path of config.json
            infer_yml (str): path of infer_cfg.yml
        """
        super().__init__(config, infer_yml)
        self.model_dir = model_dir
        self.model_file = None
        self.params_file = None
        for file in os.listdir(model_dir):
            if file.endswith(".pdmodel"):
                self.model_file = file
            elif file.endswith(".pdiparams"):
                self.params_file = file
        assert self.model_file is not None, "pdmodel file does not exsit."
        assert self.params_file is not None, "pdiparams file does not exist."

    def load(self):
        """load model"""
        paddle.enable_static()
        self.exe = fluid.Executor(fluid.CPUPlace())
        [self.paddle_prog, feed, self.fetch] = fluid.io.load_inference_model(
            self.model_dir,
            self.exe,
            model_filename=self.model_file,
            params_filename=self.params_file,
        )

    def predict(self, input_dict):
        """predict"""
        res = self.exe.run(self.paddle_prog, feed=input_dict, fetch_list=self.fetch)
        return res[0]
    
    def predict_image(self, img):
        """predict image"""
        inputs = self.preprocessor(img)
        inputs = {"image": inputs}
        res = self.predict(inputs)
        results = self.postprocessor(img, res)
        return results
    
def argsparser():
    """
    parse command arguments
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./model/config.json",
        help=("path of deploy config.json"),
    )
    parser.add_argument(
        "--infer_yml",
        type=str,
        default="./model/infer_cfg.yml",
        help=("Path of infer_yml.Detection config comes first"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help=("path of pdmodel and pdiparams"),
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="./test_images",
        help="Dir of test image file",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output_dir", help="output dir."
    )
    return parser


def main():
    """main"""
    # check output_dir
    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # get ppnc results
    config = FLAGS.config

    with open(FLAGS.infer_yml, "r") as f:
        infer_yml = yaml.safe_load(f)

    
    m = KeyPointDetector(config, infer_yml)
    
    model_dir = FLAGS.model_dir
    paddle_infer = PaddleInfer(model_dir, config, infer_yml)
    paddle_infer.load()
    
    
    path = FLAGS.test_dir
    if not os.path.exists(os.path.join(output_dir, "paddle_result_images")):
        os.makedirs(os.path.join(output_dir, "paddle_result_images"))
    if not os.path.exists(os.path.join(output_dir, "ppnc_result_images")):
        os.makedirs(os.path.join(output_dir, "ppnc_result_images"))

    if not os.path.exists(os.path.join(output_dir, "ppnc_result_pickle")):
        os.makedirs(os.path.join(output_dir, "ppnc_result_pickle"))
    if not os.path.exists(os.path.join(output_dir, "paddle_result_pickle")):
        os.makedirs(os.path.join(output_dir, "paddle_result_pickle"))

    for i in os.listdir(path):
        print(i)
        img = cv2.imread(os.path.join(path, i))

        time0 = time.time()
        res1 = m.predict_image(img)
        time1 = time.time()
        print(f'ppnc time: {time1 - time0}')
        cv2.imwrite(
            os.path.join(os.path.join(output_dir, "paddle_result_images"), i),
            VisKeypointDetection(img, res1),
        )
        with open(
            os.path.join(
                os.path.join(output_dir, "paddle_result_pickle"),
                i.split(".")[0] + ".pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(res1, file)
        time2 = time.time()
        res2 = paddle_infer.predict_image(img)
        time3 = time.time()
        print(f'paddle time: {time3 - time2}\n')

        cv2.imwrite(
            os.path.join(os.path.join(output_dir, "ppnc_result_images"), i),
            VisKeypointDetection(img, res2),
        )
        with open(
            os.path.join(
                os.path.join(output_dir, "ppnc_result_pickle"),
                i.split(".")[0] + ".pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(res2, file)


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()
    main()
