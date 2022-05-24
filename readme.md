# update
at 2022-05-24
add s3 file download

1. conda install awscli -y or pip install awscli
2. aws configure
AWS Access Key ID [None]: seafog
AWS Secret Access Key [None]: "defaultpassword" hint: \w{3}\d{7}
Default region name [None]: ENTER
Default output format [None]: ENTER

3. aws configure set default.s3.signature_version s3v4
4. aws --endpoint-url http://oldgpu:9000 s3 cp s3://seafog data --recursive


# install
- make install
환경 설치

# TRAIN
- conda activate venv/
activate 환경

- python src/step_1_pre_train_ml.py
data/ 밑에 필수 폴더 생성

- CUDA_VISIBLE_DEVICES='mygpu' python src/step_2_train_hpo_ml.py
hpo 과정. 
CUDA_VISIBLE_DEVICES 안하면 default 0번으로 돌아감

- python src/step_3_train_refit_ml.py
최적 하이퍼파라미터로  train 셋을 train = train + val 으로 업데이트한 후 한번 더 학습한 후 
모델 저장
모델 불러와서 X_test, y_test 예측
성능은 config파일에 catalogue.test_result에 있음
