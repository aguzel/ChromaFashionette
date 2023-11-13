[DOWNLOAD PAPER](https://github.com/aguzel/ChromaFashionette/blob/main/Designing%20Efficient%20Image-to-Image%20Translation%20Artificial%20Neural%20Network%20Model%20For%20Segmenting%20Fashion%20Images%20Guzel%20et%20al%202023.pdf?raw=true)

# ChromaFashio[net]te :dress: :brain:
**Designing Efficient Image-to-Image Translation Artificial Neural Network Model For Segmenting Fashion Images**

ChromaFashionette is a name for the artificial neural network built during my University of Leeds AI MSc thesis.
It combines the word "chroma," which refers to color, with " **fashio[net]** te," which is fitting for a network that segments fashion images, and the ending "-ette" that denotes smallness or subtlety (efficient and small).

![image](https://user-images.githubusercontent.com/46696280/216799057-8225705b-b6be-4854-bfc9-6a9b33ef9886.png)


![image](https://github.com/aguzel/ChromaFashionette/assets/46696280/5154c825-1834-470e-9ef9-9b40ae5e37fb)


### Install dependencies
```shell
pip3 install -r requirements.txt
```

### Dataset

```shell
--data
  -train
   * A
   * B
  -test
   * A
   * B
```

### Run training 
```shell
python3 train.py
python3 train_weighted.py
```

### Run testing
```shell
python3 test.py
```

### Training Settings

```shell
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NORMALIZE = False
ARCHITECTURE = 'DeepLabV3+'
NUM_CLASSES = 5 
LR = 1e-4
EPOCHS = 5
```

