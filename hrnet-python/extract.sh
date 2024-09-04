#! /bin/bash

model_dir="./hrnet"
rm -rf $model_dir"/__pycache__"
require="./requirements.txt"
zip_name="hrnet_deploy.zip"
zip -r $zip_name $model_dir $require -x "*/__pycache__/*"
echo "successfully extract deploy packages to $zip_name"
