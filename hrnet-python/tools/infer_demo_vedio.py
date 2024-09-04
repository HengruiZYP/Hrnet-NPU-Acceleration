import os, sys

upper_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(upper_level_path)
from hrnet import KeyPointDetector, VisKeypointDetection
import yaml
import cv2
import numpy as np
import argparse
import time

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
    
    cap = cv2.VideoCapture(0)
    # 获取摄像头的输入大小
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("w: ", width)
    print("h: ", height)
    
    if cap.isOpened():
        print("video opened")
        
    else:
        print("video not opened")
        return
        
    # 帧率参数    
    prev_time = time.time()
    frame_count = 0
    fps = 0
    
    while True:
    
        ret, image = cap.read()
        
        if not ret:
            break
        
        # 计算帧率        
        current_time = time.time()        
        frame_count += 1        
        elapsed_time = current_time - prev_time
        
        if elapsed_time > 1:            
            fps = frame_count / elapsed_time            
            prev_time = current_time            
            frame_count = 0        
               
        # 在帧上绘制帧率        
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),2)
                    
        # with_profile
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
            
        # visualize
        visualize = args.visualize
        if visualize:
            render_img = VisKeypointDetection(image, res, 0.6)
            cv2.imshow('camera', render_img)
            
        if cv2.waitKey(5) & 0xff == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argsparser()
    args = parser.parse_args()
    main(args)
