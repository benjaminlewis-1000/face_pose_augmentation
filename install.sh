#! /bin/bash

git clone --recurse-submodules https://github.com/hhj1897/face_pose_augmentation.git
pip install -e . 

mkdir tmp
cd tmp
git clone https://github.com/hhj1897/face_detection
git clone https://github.com/hhj1897/face_alignment

cd face_detection
git lfs pull
pip install -e .

cd ..
cd face_alignment
pip install -e .

cd ../.. # Back to root
mv tmp/face_detection/ibug/face_detection ibug/face_detection
mv tmp/face_alignment/ibug/face_alignment ibug/face_alignment
# rm -rf tmp