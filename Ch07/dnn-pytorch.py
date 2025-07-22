from torchvision.datasets import FashionMNIST
from sklearn.model_selection import train_test_split


fm_train = FashionMNIST(root='.', train=True, download=True)    # 저장될 위치, 훈련/테스트 선택, 다운로드해 로컬에 저장
fm_test = FashionMNIST(root='.', train=False, download=True)
type(fm_train.data) # 데이터는 객체의 data 속성에 PyTorch Tensor로 저장됨     # Tensor: PyTorch의 기본 데이터 구조
print(fm_train.targets.shape, fm_test.targets.shape)  # 타깃은 1차원 배열(원-핫 인코딩 X)

train_input = fm_train.data
train_target = fm_train.targets
train_scaled = train_input / 255.0  # 입력 정규화
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)