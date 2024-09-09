# 一、模型说明
该部署代码用于姿态检测任务，基于PaddleDetection2.5进行模型训练，paddle版本使用2.4.0

# 二、效果预览
可视化结果:

<img src="vedio/hrnet.gif" width="40%">

在经过NPU加速之后，推理时间缩短了45倍左右

如下图所示 运行test.py进行对比

paddle是CPU推理，ppnc是NPU推理

<img src="vedio/diff.png" width="30%">

# 三、使用方式
## 3.1 模型训练

本库已含训练好的模型，并进行了加速处理。

__模型生产基于aistudio平台进行__，确保已有aistudio账号。

### 3.1.1 环境准备

aistudio创建项目, 选择paddle2.4.0版本。

### 3.1.2 模型训练、评估、导出
模型生产过程请参考项目：[aistudio项目链接](https://aistudio.baidu.com/projectdetail/7161083?contributionType=1&sUid=1318783&shared=1&ts=1701160332830)



__请参考如下版本__：
![](hrnet-python/res/aistudio_version.jpg)

模型训练完成得到model.pdmodel和mode.pdiparams模型文件。

## 3.2 模型转换

### 3.2.1 已转换模型
本项目已转换好使用coco数据集训练的Hrnet模型，置于model文件夹内供使用。
### 3.2.2 其他模型
若需要转换其他自行训练的模型，请联系百度技术支持同学：ext_edgeboard01@baidu.com


## 3.3 模型部署
__模型部署基于板卡进行__

### Step1：
- 安装opencv依赖库及EdgeBoard DK-1A推理工具PPNC(如已安装，可跳过此步)
  打开终端，执行以下命令安装PPNC。
  ```bash
  sudo apt update
  sudo apt install libopencv-dev -y
  sudo apt install python3-opencv -y
  sudo apt install ppnc-runtime -y
  ```
- 安装PaddlePaddle(如已安装，可跳过此步)
  打开终端，执行以下命令安装PaddlePaddle。
  ```bash
  mkdir Downloads
  cd Downloads
  wget https://bj.bcebos.com/pp-packages/whl/paddlepaddle-2.4.2-cp38-cp38-linux_aarch64.whl 
  sudo pip install paddlepaddle-2.4.2-cp38-cp38-linux_aarch64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

### Step2：
下载项目文件
```bash
cd /home/edgeboard/
#下载模型
git clone https://github.com/HengruiZYP/Hrnet-NPU-Acceleration.git
```

### Step3：
安装依赖库，确保当前位于/home/edgeboard/Hrnet-NPU-Acceleration/hrnet-python目录下：
```bash
sudo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
### Step4：
- 配置config.json文件（无更改可略过）
  终端输入以下命令，进入config.json所在目录，并使用vim查看内容。
  ```bash
  cd /home/edgeboard/Hrnet-NPU-Acceleration/hrnet-python/
  vim config.json
  ```
- 默认已配置完成，可直接使用，如有自定义，可另行更改配置内容。
  ```json
  {
      "mode": "professional",
      "model_dir": "./model", 
      "model_file": "model"
  }
  ```

    - mode: 固定为"professional"
    - model_dir：传输至板卡的模型文件(model.json、model.nb、model.onnx、model.po)的目录
    - model_file: 传输至板卡的四个模型文件的文件名，固定为model
  
  键盘输入“:q”，回车退出。

### Step6：
尝试ppnc推理
- 运行推理脚本
    确保当前位于/home/edgeboard/Hrnet-NPU-Acceleration/hrnet-python/目录下：

    ```shell
    sudo python3 tools/infer_demo.py \
            --config ./model/config.json \
            --infer_yml ./model/infer_cfg.yml \
            --test_image test_images/0.jpeg \
            --visualize \
            --with_profile
    ```

    命令行选项参数如下：

        - config: 上文建立的config.json路径
        - infer_yml: Detection套件模型导出所带的yml文件
        - test_image: 测试图片路径
        - visualize: 是否可视化，若置为True，则会在该路径下生成vis.jpg渲染结果
        - with_profile: 是否统计前处理、模型推理、后处理各阶段耗时，若置为True，会输出各阶段耗时

  
- 查看工程目录，若得到结果如下：vis.jpg
  
  证明模型已经部署成功
  
  <img src="hrnet-python/vis.jpg" width="30%">

### Step7：
接入摄像头识别
- 插入摄像头
  ```bash
  ls /dev/video*
  #查看摄像头是否正常
  ```
  正常情况应该如下：
  
  <img src="vedio/image.jpeg" width="40%">
  
- 摄像头的代码在infer_demo_vedio.py，输入如下指令，即可开启摄像头识别
  ```bash
  cd /home/edgeboard/Hrnet-NPU-Acceleration/hrnet-python/
  sudo python3 tools/infer_demo_vedio.py --visualize



## 3.4 paddle、ppnc对比测试
- 模型文件 
    首先将模型生产阶段产生的xx.pdmodel，xx.pdiparams模型文件传输至板卡，置于./model文件夹

- 执行测试

    确保当前位于/home/Hrnet-NPU-Acceleration/edgeboard/hrnet-python目录下：

    ```shell
    sudo python3 tools/test.py \
    --config  ./model/config.json  \
    --infer_yml ./model/infer_cfg.yml  \
    --model_dir ./model \
    --test_dir ./test_images \
    --output_dir ./output_dir
    ```

        各命令行选项参数如下：
        - config: 同上
        - infer_yml: 模型导出时生成的infer_cfg.yml文件
        - model_dir: paddle静态图模型文件(model.pdmodel)和(model.pdiparams)所在目录
        - test_dir: 测试图片文件夹路径
        - output_dir: 存放结果文件，默认为"./output_dir"，该路径下会生成paddle_result_pickle、paddle_result_images、ppnc_result_pickle、ppnc_result_images目录，分别存放paddle和ppnc的pickle格式结果和可视化的结果数据。

## 3.5 实际项目部署
实际用于项目中时，仅需要部分脚本，因此需要提取部署包并置于实际的项目代码中运行。

### 3.5.1 提取部署包
确保当前位于/home/Hrnet-NPU-Acceleration/edgeboard/hrnet-python, 执行以下命令导出用于项目部署的zip包：
```shell
sudo ./extract.sh
```
执行成功后会在当前目录生成hrnet_deploy.zip压缩包。

### 3.5.2 使用部署包
- 准备ppnc模型及配置文件  
    将模型转换阶段生成的model.nb、model.json模型文件拷贝到项目能访问的目录，并参照3.3.1的方式编写模型配置文件config.json。

- 准备环境   
    将3.3.3.1生成的hrnet_deploy.zip部署包解压后得到hrnet文件夹和requirements.txt文件。其中requirements.txt是已验证过部署包可正常使用的相关库版本，实际项目开发中安装相关库时可参考该文件。
- 使用   
    部署包使用方式请参考[3.3.1-ppnc推理]中使用的infer_demo.py脚本


