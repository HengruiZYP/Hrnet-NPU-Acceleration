import os, sys

upper_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(upper_level_path)
from hrnet import KeyPointDetector, VisKeypointDetection
import yaml
import cv2
import numpy as np
import argparse


def argsparser():
    """
    argparser
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="./model/config.json",
        type=str,
        help=("Path of deploy config.json.Detection config comes first"),
    )
    parser.add_argument(
        "--infer_yml",
        default="./model/infer_cfg.yml",
        type=str,
        help=("Path of infer_yml.Detection config comes first"),
    )
    parser.add_argument(
        "--test_image",
        type=str,
        default="./test_images/0.jpeg",
        help="Path of test image file.",
    )
    parser.add_argument(
        "--visualize", action="store_true",help="whether to visualize."
    )
    parser.add_argument(
        "--with_profile",
        action="store_true",
        help="whether to predict with profile.",
    )
    return parser


def main(args):
    """main"""
    
    # init PPNCDetector
    deploy_config = args.config
    assert os.path.exists(deploy_config), "config does not exist."
    
    assert os.path.exists(args.infer_yml), "infer_yml does not exist."
    with open(args.infer_yml, "r") as f:
        infer_yml = yaml.safe_load(f)
    
    net = KeyPointDetector(deploy_config, infer_yml)
    
    image_path = args.test_image
    assert os.path.exists(image_path), "test_image does not exist."
    image = cv2.imread(image_path)
    
    with_profile = args.with_profile
    if with_profile:
        # return with time consumption for each stage
        res = net.predict_profile(image)
        print("preprocess time: ", net.preprocess_time)
        print("predict time: ", net.predict_time)
        print("postprocess time:", net.postprocess_time)

        total_time = (
            net.preprocess_time + net.predict_time + net.postprocess_time
        )
        print("total time: ", total_time)
    else:
        res = net.predict_image(image)
    
    visualize = args.visualize
    if visualize:
        render_img = VisKeypointDetection(image, res, 0.6)
        cv2.imwrite("./vis.jpg", render_img)
        print("visualize result saved as vis.jpg.")
    


if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    main(args)
