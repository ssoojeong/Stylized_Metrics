# Stylized_Metrics

## 250209_Added
- CLIP Similarity  Loss
- VGG Content Loss
- VGG Style Loss


## All in One 
modify the argparse components
```
CUDA_VISIBLE_DEVICES=0 python allinone.py
```
## FID
### how to use
```
python fid_score.py ../clic22val /root/workspace/ImageQuality/IQT-main/experiment_clic22val/elic_char2e6_lp06_styl5e1_gan1_face3e3_0016/815/ --device cuda --batch-size 1
```

## KID
### link
https://github.com/abdulfatir/gan-metrics-pytorch
### how to use
```
python kid_score.py --true ../clic22val  --fake /root/workspace/ImageQuality/IQT-main/experiment_clic22val/elic_char2e6_lp06_styl5e1_gan1_face3e3_0016/815/ --batch-size 1
```
## LPIPS
### link
### how to use
python lpips_score.py

## DISTS
### link
https://github.com/dingkeyan93/DISTS
### how to use
```
python DISTS_pt.py --ref /root/workspace/ImageQuality/kodak/ --dist /root/workspace/ImageQuality/IQT-main/experiment/elic_char2e6_lp1_sty1e2_gan1_0016/900/ --exp elic_char2e6_lp1_sty1e2_gan1_0016
```
## IQT
### link
https://github.com/anse3832/IQT
### how to use
```
python iqt_test.py
```
