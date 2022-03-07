# MSBD5001 Foundations of Data Analytics

MSBD5001 Foundations of Data Analytics, Spring 2022. All rights reserved by Lam Chun Ting Jeff

Dataset from
https://www.kaggle.com/c/msbd5001-spring-2022/data

## Usage
- Training a model
```python main.py train```

Optional Parameter
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|--train-path TRAIN_PATH | train.csv | Training dataset path e.g. train.csv
|--model MODEL        | model.obj | Model directory e.g. model.obj


- Testing a model
```python main.py test```

Optional Parameter
| Parameter                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
|--test-path TEST_PATH | test.csv | Testing dataset path e.g. test.csv
|--model MODEL | model.obj |         Model directory e.g. model.obj
|--output SUBMISSION_PATH |  submission.csv |  Submission output path e.g. submission.csv