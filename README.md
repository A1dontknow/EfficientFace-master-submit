# EfficientFace

*Zengqun Zhao, Qingshan Liu, Feng Zhou. "[Robust Lightweight Facial Expression Recognition Network with Label Distribution Training](https://drive.google.com/file/d/1yDpyQ1emZ8IObPNZt76ljeW98GP-Dw22/view?usp=sharing)". AAAI'21*

## Requirements

- Python >= 3.6
- PyTorch >= 1.2
- torchvision >= 0.4.0
- seaborn
- grad-cam

## Training

- Step 1: Download [RAF-DB](https://drive.google.com/file/d/1XkZu0p406YqKZLtsGtgVI-ZDOhKJMq6-/view?usp=sharing), and make sure it has the structure like the following:

```
./RAF-DB/
         train/
               0/
                 train_09748.jpg
                 ...
                 train_12271.jpg
               1/
               ...
               6/
         test/
              0/
              ...
              6/

[Note] 0: Neutral; 1: Happiness; 2: Sadness; 3: Surprise; 4: Fear; 5: Disgust; 6: Anger
```

- Step 2: download pre-trained model from [Google Drive](https://drive.google.com/file/d/1sRS8Vc96uWx_1BSi-y9uhc_dY7mSED6f/view?usp=sharing), and put it into ***./checkpoint***.
- Step 3: run ```train.py ```


## Quick test, CAM, visualization:

- Step 1: Download [model](https://drive.google.com/file/d/1i1voKeEMHvzv6Z4pa9uporgUcYOstFIQ/view?usp=sharing), and put it into ***./checkpoint***.) 
- Step 2: Run 1 of following:
  - ```eval.py```: Test and display confusion matrix on RAF-DB test set (make sure you download the [RAF-DB](https://drive.google.com/file/d/1XkZu0p406YqKZLtsGtgVI-ZDOhKJMq6-/view?usp=sharing)).
  - ```test&vis/CAM.py```: Generate CAM on arbitrary image (remember to change the path in the code. Default is [RAF-DB](https://drive.google.com/file/d/1XkZu0p406YqKZLtsGtgVI-ZDOhKJMq6-/view?usp=sharing) test picture)
  - ```test&vis/classify.py```: Classify all pictures in ***./test&vis/test_data***.
  - ```test&vis/visualization.py```: Same as ```test&vis/classify.py```, but with visualization.

