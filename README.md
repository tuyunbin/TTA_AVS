# 基于视觉标签和词性信息引导的视频内容自动语言描述技术研究

This code is tested on python 3.6 and pytorch 1.3.0

## Setup:
Install all the required packages from requirements.txt file.

#### Datasets:
Download the ResNet-152 frame-level features + ResNeXt-101 motion features for [MSR-VTT](https://drive.google.com/file/d/1bZJ0noxJ9EwXV161d4w6p6PqaszhM8t4/view?usp=sharing) videos.

Download the captions and vocabulary data for [MSR-VTT](https://drive.google.com/drive/folders/1HhF8Tl3ZXQzaILlg6vCXST9A_6BjPU5r?usp=sharing), and place the downloaded data in 'data' folder. 

### Evaluation:
Clone the code from *[here](https://github.com/ramakanth-pasunuru/video_caption_eval_python3)* to setup the evaluation metrics, and place it the parent directory on this repository. Note that this is also required during training since tensorboard logs the validation scores.


## Run Code:
To train Baseline-XE model:
```
python main.py --model_name "model_name"
```
```
For testing:
```
python main.py --mode test --load_path "path_to_model_folder" --beam_size 5 
