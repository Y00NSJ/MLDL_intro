from torchvision.datasets import FashionMNIST


fm_train = FashionMNIST(root='.', train=True, download=True)    # 저장될 위치, 훈련/테스트 선택, 다운로드해 로컬에 저장
fm_test = FashionMNIST(root='.', train=False, download=True)
type(fm_train.data) # 데이터는 객체의 data 속성에 PyTorch Tensor로 저장됨
# Tensor: PyTorch의 기본 데이터 구조
