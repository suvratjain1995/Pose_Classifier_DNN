# Pose Classifier using DNN
This is an extended version of  [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) project  

## Introduction


This is a new improved version. The main objective was to remove
dependency on separate c++ server which besides the complexity
of compiling also contained some bugs... and was very slow.
The old version utilizing [rmpe_dataset_server](https://github.com/michalfaber/rmpe_dataset_server) is
still available under the tag [v0.1](https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation/releases/tag/v0.1) if you really would like to take a look.

## Testing steps
- Convert caffe model to keras model or download already converted keras model https://www.dropbox.com/s/llpxd14is7gyj0z/model.h5
- Run the notebook `demo.ipynb`.
- `python demo_image.py --image sample_images/ski.jpg` to run the picture demo. Result will be stored in the file result.png. You can use
any image file as an input.

## Training steps


- Install gsutil `curl https://sdk.cloud.google.com | bash`. This is a really helpful tool for downloading large datasets. 
- Download the data set (~25 GB) `cd dataset; sh get_dataset.sh`,
- Download [COCO official toolbox](https://github.com/pdollar/coco) in `dataset/coco/` . 
- `cd coco/PythonAPI; sudo python setup.py install` to install pycocotools.
- Go to the "training" folder `cd ../../../training`.
- Optionally, you can set the number of processes used to generate samples in parallel
  `dataset.py` -> find the line `df = PrefetchDataZMQ(df, nr_proc=4)`
- Run the command in terminal `python train_pose.py`

## Related repository
- CVPR'16, [Convolutional Pose Machines](https://github.com/shihenw/convolutional-pose-machines-release).
- CVPR'17, [Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation).

