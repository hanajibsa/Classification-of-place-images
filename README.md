# Classification-of-place-images

# **1. Introduction**

본 프로젝트는 train station, farm, campus에 해당하는 각각의 이미지 3000장, 총 9000장의 이미지에 대하여 최적의  CNN(convolutional neural network)  분류 모델을 도출하는 것을 목표로 한다.

이를 위해 다음과 같은 parameter들을 변경시키는 과정을 거쳤다.

- Data Augmentation
- CNN Model Selection
- Optimizer, Loss Function, Learning rate(+Scheduler), Regularization, Batch size

# **2. Data Preparation**

## **2.1 Dataset split**

총 9000개의 데이터셋을 train: test = 8:2로 나누어 7200개의 train set, 1800개의 test set을 만들었다.

## **2.2 Resize**

후에 최종 모델로 선정한 VGG16의 input image size인 (224,224)에 맞추기 위해 resize를 적용하였다.

## **2.3 Data Augmentation**

Underfitting을 방지하기 위해 데이터의 개수를 늘리고자 train set에 대해 RandomHorizontalFlip을 적용하였다. p=1로 설정하여 7200개의 Horizontal Flip된 이미지를 생성하였고, 이를 기존 train set에 concatenate하여 총 14400개의 train dataset을 형성하였다.

# **3. Model Selection**

## **3.1 Model**

epoch = 50, Ir = 0.0001, optimizer = Adam, batch_size = 64의 조건에서 VGG16, ResNet34, AlexNet, GoogLeNet를 비교하였다.

**VGG16**

<img width="80%" src="https://github.com/hanajibsa/Classification-of-place-images/assets/115071643/5fb4cf2a-2097-4dde-b338-ea202552602d"/>


(99.0486, 83.16667)

**ResNet34**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e84be54f-8e88-4a01-b7e8-376c119dca19/Untitled.png)

(99.4722, 85.0556)

**AlexNet**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8779813e-62ee-4dda-a106-f255d9158388/Untitled.png)

(91.8194, 77.1111)

**GoogleNet**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/dd07d547-8ba6-47d1-b2a8-fe828d8e8884/Untitled.png)

(97.5444, 78.08)

(train accuracy, validation accuracy)이며, 소수점 아래 다섯번째 자리에서 반올림하였다.

실행 결과 VGG와 ResNet34의 성능이 가장 우수했지만 ResNet34은 epoch 초반부터 oscillation현상이 심하게 나타났기 때문에 최종 모델로 VGG를 선정하였다.

## **3.2 Optimizer**

VGG16의 경우 epoch 20 전에 정확도가 saturation되는 것을 확인하였기 때문에 epoch = 25, Ir = 0.0001, batch_size = 64의 조건에서 SGD, Adam, RMSProop, Nadam를 비교하였다.

**Adam**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1653d9f5-4eab-4787-9f2f-bc11a825e599/Untitled.png)

(98.6458,83.3889)

**NAdam**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/689a156b-11c7-4efa-964a-6e56dac315eb/Untitled.png)

(98.8958,83.1111)

**SGD**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/b9586d84-0e89-4732-98ef-839776328e17/Untitled.png)

(36.2292,40.2778)

**RMSProp**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e58f1959-adc5-49d1-8535-4f4adcfb0094/Untitled.png)

(62.9792,79.5556)

(train accuracy, validation accuracy)이며, 소수점 아래 다섯번째 자리에서 반올림하였다.

가장 우수한 성능을 보인 Adam과 Nadam 중, validation error가 적은 Adam을 최종 Optimizer로 선정했다.

## **3.3 Loss Function**

다중 클래스 분류 작업에 주로 활용되는 Cross Entropy Loss 함수를 선택했다. 이 손실 함수는 모델의 성능 향상을 위해 손실을 최소화하기 위한 가중치 조정 및 업데이트를 자동으로 수행하며, Gradient 계산에 있어서도 효율적으로 처리할 수 있어 모델의 가중치 업데이트를 신속하고 정확하게 수행할 수 있다. 따라서, 본 프로젝트에서는 Cross Entropy Loss 함수를 활용하여 결과를 도출했다.

## **3.4 Learning rate**

앞선 결과에서 overfitting이 발생한 것을 확인하였다. 이러한 문제를 해결하기 위해 learning rate를 조절하였다.

**lr = 1e-5**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/50690786-361f-4a3f-90d8-49969f11b327/Untitled.png)

(96.0417, 82.7778)

**CosineAnnealingWarmRestarts**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4350004f-2bdf-4a4d-a1f1-e24fb9151565/Untitled.png)

(98.5833, 81.3333)

**LAMBDA LR**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c319d2d3-60f6-4242-ad20-d6e146378c57/Untitled.png)

(99.0833, 83.1111)

**StepLR**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/641a4c35-bd8d-41cc-b284-c8bb0ac0f06a/Untitled.png)

(35.1138, 35.5716)

(train accuracy, validation accuracy)이며, 소수점 아래 다섯번째 자리에서 반올림하였다.

Iearning rate를 1e-5로 줄였을 때 학습 속도가 느려져 정해진 epoch 내에서 최댓값을 달성하지 못했다. learning scheduler를 적용한 결과, 최대의 정확도를 도출한 scheduler는 LAMBDA LR이었으나 scheduler를 적용하지 않은 것에 비해 유의미한 성능 향상이 생기지 않아 적용하지 않기로 결정했다.

## **3.5 Regularization**

overfitting을 방지하고 모델의 generalization ability를 높이기 위해 weight decay를 적용하였다.

**lr = 0.0001, weight_decay=0.01**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/67598293-bb63-4d52-b284-2cb9a4bea72e/Untitled.png)

**lr = 0.0001, weight_decay=1e-5**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7a4fe2c7-e406-4d5c-becd-8163a8b1675b/Untitled.png)

(98.7847, 84.5)

(train accuracy, validation accuracy)이며, 소수점 아래 다섯번째 자리에서 반올림하였다.

가장 먼저 0.001과 0.0001의 두가지 Learning rate를 수행한 결과, Ir = 0.001인 경우 지속적인 underfitting이 발생하여 더 이상 성능이 향상 되지 않을 것이라 판단하여 학습을 중단했다. 이후 Ir = 0.0001로 설정하고, weight decay 를 각각 0.01 과 1e-5로 설정하여 수행한 결과 weight decay = 0.001인 경우 또한 계속해서 underfitting이 발생하여 학습을 중단했다. 결과적으로 가장 유의미한 정확도를 도출한 Ir = 0.0001과 weight decay = 1e-5를 최종 하이퍼 파라미터 값으로 선택했다.

## **3.6 Batch size**

optimizer= Adam, Ir = 0.001을 고정으로 설정한 모델에 대하여  batch size를 16, 32, 48, 64로 변경해가며 정확도를 비교하였다. 학습 결과 batch size는 정확도에 있어 큰 차이를 보이지는 않았으나, 그 중 train, validation accuracy가 가장 높았던 64를 최종적으로 선택하였다.

**16**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b465268-ba68-43a4-8a88-539a186693cd/Untitled.png)

(98.4028, 84.2778)

**32**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/595f7378-4c6f-4881-8e62-576752b7a5b6/Untitled.png)

(98.6597, 80.8889)

**48**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d32b374c-bd41-4bb4-ba15-53702292ff9e/Untitled.png)

(98.5720, 83.2594)

**64**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/62d41f1a-f789-40ad-9a86-7e5aaf9a2d93/Untitled.png)

(98.7847, 84.5)

(train accuracy, validation accuracy)이며, 소수점 아래 다섯번째 자리에서 반올림하였다.

# **4. Result**

## **4.1 Model**

최종적으로 선택한 모델은 VGG16으로 3x3 convolution layer 두 층과 maxpooling 층을 총 다섯 번 거친 후 fully connected layer로 들어가는 구조이다. convolution layer를 거친 후 ReLU function을 적용하여 기울기 소실 문제를 해결하고, 학습 속도를 높였으며 fully connected layer 블럭의 Dropout을 통해 overfitting을 방지할 수 있다는 장점이 있다. 하이퍼파라미터 튜닝을 통해 optimizer = Adam, learning rate = 0.0001, weight decay = 1e-5, epoch = 25로 설정하였다.

https://lh6.googleusercontent.com/LlNVDKiQxVBzSImZAL3MbzIiNZXbB5eYRd8YLLvzBwC1_QYJ_g45oF0L1ejioIqCyLTulAuXNJv4eoGNijkc3aKPF7qYBllq-tiNhfmc1DV8dlCyJd3QJBxYC3I9Ek3i0Ml77YQLON5G81ZeFBZKoVo

## **4.2 Accuracy**

최종 모델의 train accuracy는 98.5833, validation accuracy는 83.5이었다. validation accuracy 그래프를 보면, 초기 epoch 3 주변까지는 정확도가 빠르게 증가하다가 그 이후에 약간의 진동을 보이며, 결과적으로 우수한 정확도에 도달하는 것을 확인할 수 있었다.  이는 일반적으로 초기에는 모델이 데이터에 적응하며 학습되기 시작하므로 정확도가 높은 값을 띄며, 진동은 학습 과정에서 모델이 미세한 패턴을 학습하거나 최적화하는 과정으로 해석된다.

https://lh5.googleusercontent.com/NxWPvD1YttRh-T_Eir-68QN5GKMGizoWak1b91-NsojUEmOdpHrEk6xNs7IQ0GLh62CYO5dvZIDQdl_CSJaxpxEl7lDzg0c5eJ7FPOcF_HY0B0pTB4s0rWaa9f023-ZC1VTopW7hQAuvyA7gFqprTQ8

## **4.3 Performance Evaluation**

최종 모델에 주어진 30장의 test_sample 이미지를 입력값으로 넣었을 때, 다음과 같이 분류되었다.
