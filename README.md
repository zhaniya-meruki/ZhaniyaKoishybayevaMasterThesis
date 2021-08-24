This repository was adapted from https://github.com/Rudrabha/Lip2Wav
# Lip2Wav

*Generate high quality speech from only lip movements*. This code is part of the paper: _Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis_ published at CVPR'20.

[[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Prajwal_Learning_Individual_Speaking_Styles_for_Accurate_Lip_to_Speech_Synthesis_CVPR_2020_paper.pdf) | [[Project Page]](http://cvit.iiit.ac.in/research/projects/cvit-projects/speaking-by-observing-lip-movements) | [[Demo Video]](https://www.youtube.com/watch?v=HziA-jmlk_4)
 <p align="center">
  <img src="images/banner.gif"/></p>

----------
Highlights
----------
 - First work to generate intelligible speech from only lip movements in unconstrained settings.
 - Sequence-to-Sequence modelling of the problem.
 - Dataset for 5 speakers containing 100+ hrs of video data made available! [[Dataset folder of this repo]](https://github.com/Rudrabha/Lip2Wav/tree/master/Dataset) 
 - Complete training code and pretrained models made available.
 - Inference code to generate results from the pre-trained models.
 - Code to calculate metrics reported in the paper is also made available.

### You might also be interested in:
:tada: Lip-sync talking face videos to any speech using Wav2Lip: https://github.com/Rudrabha/Wav2Lip

Prerequisites
-------------
- `Python 3.7.4` (code has been tested with this version)
- ffmpeg: `sudo apt-get install ffmpeg`
- Install necessary packages using `pip install -r requirements.txt`
- Face detection [pre-trained model](https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth) should be downloaded to `face_detection/detection/sfd/s3fd.pth`. Alternative [link](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8) if the above does not work.

Preprocessing the dataset
----------
```bash
python preprocess.py --speaker_root Dataset/subject_folder --speaker subject
```

Additional options like `batch_size` and number of GPUs to use can also be set.


Generating for the given test split
----------
```bash
python complete_test_generate.py -d Dataset/subject_folder -r Dataset/subject_folder/test_results \
--preset synthesizer/presets/subject_folder.json --checkpoint <path_to_checkpoint>

#A sample checkpoint_path  can be found in hparams.py alongside the "eval_ckpt" param.
```

This will create:
```
Dataset/subject_folder/test_results
├── gts/  (cropped ground-truth audio files)
|	├── *.wav
├── wavs/ (generated audio files)
|	├── *.wav
```

Calculating the metrics
----------
You can calculate the `PESQ`, `ESTOI` and `STOI` scores for the above generated results using `score.py`:
```bash
python score.py -r Dataset/subject_folder/test_results
```

Training
----------
```bash
python train.py <name_of_run> --data_root Dataset/subject_folder/ --preset synthesizer/presets/subject.json
```
Additional arguments can also be set or passed through `--hparams`, for details: `python train.py -h`


License and Citation
----------
The software is licensed under the MIT License. Please cite the following paper if you have use this code:
```
@InProceedings{Prajwal_2020_CVPR,
author = {Prajwal, K R and Mukhopadhyay, Rudrabha and Namboodiri, Vinay P. and Jawahar, C.V.},
title = {Learning Individual Speaking Styles for Accurate Lip to Speech Synthesis},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```


Acknowledgements
----------
The repository is modified from this [TTS repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning). We thank the author for this wonderful code. The code for Face Detection has been taken from the [face_alignment](https://github.com/1adrianb/face-alignment) repository. We thank the authors for releasing their code and models.
