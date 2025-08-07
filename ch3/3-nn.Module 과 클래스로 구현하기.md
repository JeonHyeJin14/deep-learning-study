이번에는 파이토치에서 이미 구현되어져 제공하고 있는 함수를 불러오는 것으로 더 쉽게 선형 회귀 모델을 구현

nn.Linear() : 선형회귀모델

nn.functional.mse_loss() : 평균 제곱오차

EX)
```python
import torch.nn as nn
model = nn.Linear(input_dim, output_dim)

import torch.nn.functional as F
cost = F.mse_loss(prediction, y_train)
```

# 1. 단순 선형 회귀 구현

```python
#도구 임포트
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
# y=2x를 가정된 상태에서 만들어진 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
# 하나의 입력에 대해서 하나의 출력을 가짐
# torch.nn.Linear와 같은 신경망 레이어는 내부적으로 학습 가능한 파라미터를 가지고 있음
# 초기의 가중치와 편향은 랜덤으로 설정이 됨 -> 아무 숫자나 고르는 것이 아닌 학습이 잘 되도록 설계된 범위 안에서의 무작위 숫자 
model = nn.Linear(1,1)

print(list(model.parameters()))
```
<img width="632" height="82" alt="image" src="https://github.com/user-attachments/assets/f8e5cd24-dc34-4289-829a-6fc1339e56c4" />

- 2개의 값이 출력 -> 첫 번째 = W, 두 번쨰 = b 해당 됨
- 두 값 모두 현재는 랜덤 초기화 상태 + 학습의 대상이기 때문에 requires_grad=True 설정
  
```python
# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

<img width="350" height="508" alt="image" src="https://github.com/user-attachments/assets/fda84b3a-ac72-4c0f-9cb6-c4e206e621f2" />

- cost의 값이 매우작음 -> W와 b의 값도 최적화가 되었는지 확인

```python
# 임의의 입력 4를 선언
new_var =  torch.FloatTensor([[4.0]]) 
# 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) # forward 연산
# y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
print("훈련 후 입력이 4일 때의 예측값 :", pred_y) 
```

<img width="795" height="38" alt="image" src="https://github.com/user-attachments/assets/bcfbaed3-8ef0-4822-bb60-baada730a9c8" />

- 이 문제의 정답은 y=2x이므로 어느정도 최적화 완료

# 2. 다중 선형 회귀 구현
- nn.Linear와 nn.function.mse_loss()로 다중 선형회귀 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)

print(list(model.parameters()))
```

<img width="821" height="85" alt="image" src="https://github.com/user-attachments/assets/949c95b6-9598-44cd-8c46-51b3b290fec5" />

- 3개의 가중치와 편향이 저장되어있음

- 옵티마이저 정의 -> model.parameters()를 사용해 가중치와 편향 전달

```python
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

<img width="370" height="506" alt="image" src="https://github.com/user-attachments/assets/6691a4b6-2871-42dc-b65d-be9fed1f9ac4" />

# 3. 모델을 클래스로 구현
```python
# 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
model = nn.Linear(1,1)
```

이를 클래스로 구현하면

```python
class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
    def __init__(self): #
        super().__init__()
        self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
```
- 클래스 형태의 모델은 nn.Mosule() 을 상속 받음 -> __init__()에서 모델의 구조와 동작을 정의한느 생성자 정의
  이는 파이썬에서 객체가 갖는 속성값을 초기화하는 역할로 객체가 생성될때 자동으로 호출됨
- super() 함수를 부르면 여기서 만든 클래스는 nn.Module 클래스의 속성들을 가지고 초기화 됨
  여기서 초기화는 기본값을 설정하고, 내부 기능을 사용할 준비를 하는 것
- foward() 함수는 모델이 학습 데이터를 입력 받아서 forward 연산을 진행시킴
  이 함수는 model 객체를 데이터와 함께 호출하면 자동으로 실행됨

```python
# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3,1)
```
이를 클래스로 구현하면

```python
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
```

# 4. 단순 선형 회귀 클래스로 구현하기
- 달라진 점은 모델을 클래스로 구현한 점
  
```python
# 데이터
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01) 

# 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward() # backward 연산
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

<img width="367" height="513" alt="image" src="https://github.com/user-attachments/assets/56b41831-3f28-4b1b-bfa4-94ffb30dbc98" />

1. 모델은 x_train을 사용해 예측값 계산 -> 평균 제곱 오차 함수 F.mse_loss를 사용해서 계산 함 = 비용
2. 모델이 비용을 줄이도록 옵티마이저의 기울기를 초기화
3. 비용 함수를 미분해 각 파라미터에 대한 기울기 계산
4. 기울기를 사용해 옵티마이저는 모델의 파라미터를 업데이트 -> 비용을 줄이는 방향으로 모델 개선
5. 이 과정은 설정한 횟수 2000번 만큼 반복되며 보델은 점차 더 정확한 예측을 하도록 학습됨
6. 학습이 100번 진행될 때마다 현재 Epoch 번호와 비용을 출력해 학습 과정이 어떻게 진행되고 잇는지 확인 가능
7. 로그는 학습 중에 ㅁ도ㅔㄹ의 성능이 어떻게 변화하는지를 보여주는 지표가 된다.

# 5. 다중 선형 회귀 클래스로 구현하기

```python

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1) # 다중 선형 회귀이므로 input_dim=3, output_dim=1.

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5) 

```

단순 선형 회귀 구현 코드에서 입력 차원만 바뀜

SGD 옵티마이저를 설정함

```python
nb_epochs = 2000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
    # 100번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))
```

1. 학습은 모델이 주어진 입력 데이터를 사용해 예측값을 계산
- 예측값 = 출력 , mosel(x_train)을 호출해 계산됨 -> 모델의 foward 메서드를 호출하는 것과 동일한 동작

2. 예측값이 계산된 후 , 이 값과 실제 목표값 y_train 간의 차이를 계산하는데, 이 차이를 손실 또는 비용이라고 함
- 파이토치의 F.mse_loss() 함수를 사용해 평균 제곱 오차를 계산한다.
- 이 비용은 모델이 얼마나 잘못 예측했는지를 나타냄

3. 모델이 비용을 줄이도록 학습하기 위해서 옵티마이저의 기울기 값을 초기화하고 비용 함수를 기준으로 기울기를 계산해 업데이트 함
- 이 과정은 비용 하수를 모델 파라미터에 대해 미분한 후, 옵티마이저가 이를 사용해서 파라미터를 조정하는 방식으로 이루어짐

4. 이 과정은 설정한 횟수만큼 반복되며, 모델은 점차 더 정확한 예측을 하도록 학습


