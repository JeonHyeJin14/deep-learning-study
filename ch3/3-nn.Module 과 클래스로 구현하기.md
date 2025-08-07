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
model = nn.Linear(1,1)
