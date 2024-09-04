""""hrnet"""
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .preprocess import Preprocess
from .postprocess import KptPostProcess
import numpy as np
import time
import os
import sys
upper_level_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../lib"))
sys.path.append(upper_level_path)
from libhrnet_bind import HrnetPredictor



class KeyPointDetector(object):
    """Keypoint ppnc detector"""
    def __init__(self, config, infer_yml):
        """init

        Args:
            config (str): path of config.json
            infer_yml (str): path of infer_cfg.yml
        """
        self.model = HrnetPredictor(config)
        self.preprocessor = Preprocess(infer_yml)
        self.postprocessor = KptPostProcess()
        self.prerocess_time = 0
        self.predict_time = 0
        self.postprocess_time = 0

    def predict_image(self, img):
        """predict image"""
        inputs = self.preprocessor(img)
        res = self.model.predict(inputs)
        results = self.postprocessor(img, res)
        return results

    def predict_profile(self, img):
        """predict image with profile"""
        time0 = time.time()
        inputs = self.preprocessor(img)
        time1 = time.time()
        self.preprocess_time = time1 - time0

        res = self.model.predict(inputs)
        time2 = time.time()
        self.predict_time = time2 - time1

        results = self.postprocessor(img, res)
        time3 = time.time()
        self.postprocess_time = time3 - time2
        return results