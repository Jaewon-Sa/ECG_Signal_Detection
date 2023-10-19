# Heart_Signal_Detection
The goal of this project is to create a learning system that returns location of fundamental heart sounds (S1 and S2)

## Data
To train the data, we need annotated labels for S1 and S2 in human heart sounds. We were lucky enough to get medical data from physionet.
All of it can be found [here](https://physionet.org/content/circor-heart-sound/1.0.3/)

## using model
We used two types: a mobilenetv3+SSD structure and a custom-made VIT structure using Transformer's Encoder. 

- Mnet+SSD
![image](https://github.com/Jaewon-Sa/Heart_Signal_Detection/assets/92181151/7f55eae7-c30a-479f-b327-ed69a4121e04)

- VIT

![image](https://github.com/Jaewon-Sa/Heart_Signal_Detection/assets/92181151/45ab2408-537e-4fe1-9ed1-6d0b89b3ae7a)

If the temporary VIT model performed well, we planned to learn the auscultated locations using the cls token through additional labeling,
but it is currently showing underfitting.

## Training & test
The code for the Training and testing can be found in python/example.ipynb.

- When we created our custom dataset, we started with data label 1.

## Performance
We set the train,valid,test set in the ratio 0.6,0.2,0.2 and the model performance was plucked based on the test set. 

We haven't trained on all parameters, but here's what we've seen so far.

| Type | SR | Channels | Filter | Time masking | Pretrain(ImageNet) | Aug | X(wl=sr/x) | mAP0.5 |
| :--: | :-: | :-----: | :----: | :----------: | :----------------: | :-: | :--------: | :----: |
| MnetSSD | 8000 | 1 | High, Low | ✔ | ✔ |  | 15 | 0.802 |

