# install
- make install
환경 설치

# TRAIN
- conda activate venv/
activate 환경

- CUDA_VISIBLE_DEVICES='mygpu' python src/step_2_train_hpo_ml.py
hpo 과정. 
CUDA_VISIBLE_DEVICES 안하면 default 0번으로 돌아감

- python src/step_3_train_refit_ml.py
최적 하이퍼파라미터로  train 셋을 train = train + val 으로 업데이트한 후 한번 더 학습한 후 
모델 저장
모델 불러와서 X_test, y_test 예측
성능은 config파일에 catalogue.test_result에 있음
