from data_util import DataUtil
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from argparse import ArgumentParser
import numpy as np
import traceback 
import pickle

def train(train_path, model_path):
    dataUtil = DataUtil()
    print("Reading training data from dataset/{}".format(train_path))
    X_train, Y_train = dataUtil.readTrainData(train=train_path)
    Y_train = np.ravel(Y_train)

    print("Training model... Will output model to {}".format(model_path))
    randomForest = RandomForestClassifier()
    decisionTree = randomForest.fit(X_train,Y_train)
    print("Finish training")
    pickle.dump(decisionTree, open(model_path, 'wb'))
    print("Success save model to {}".format(model_path))


def test(test_path, model_path, submission_path):
    print("Loading model {}".format(model_path))
    decisionTree = pickle.load(open(model_path, 'rb'))

    print("Reading testing data from dataset/{}".format(test_path))
    dataUtil = DataUtil()
    X_test = dataUtil.readTestData(test=test_path)

    print("Predicting model... Will output result to {}".format(submission_path))
    predict = decisionTree.predict(X_test)
    dataUtil.outputFile(predict, output=submission_path)
    print("Success output result to {}".format(submission_path))



if __name__ == "__main__":

    parser = ArgumentParser(description="MSBD5001 Individual Project")

    subparsers = parser.add_subparsers(dest="mode", required=True, help="sub commands")
    train_parser = subparsers.add_parser("train", help="Train model")
    test_parser = subparsers.add_parser("test", help="Test model")

    train_parser.add_argument("--train-path", type=str, help="Training dataset path e.g. train.csv", dest="train_path", default="train.csv")
    train_parser.add_argument("--model", type=str, help="Model directory e.g. model.obj", dest="model", default="model.obj")

    test_parser.add_argument("--test-path", type=str, help="Testing dataset path e.g. test.csv", dest="test_path", default="test.csv")
    test_parser.add_argument("--model", type=str, help="Model directory e.g. model.obj", dest="model", default="model.obj")
    test_parser.add_argument("--output", type=str, help="Submission output path e.g. submission.csv", dest="submission_path", default="submission.csv")

    opt = parser.parse_args()
    try:
        if (opt.mode == "train"):
            train(opt.train_path, opt.model)
        elif (opt.mode == "test"):
            test(opt.test_path, opt.model, opt.submission_path)
    except Exception as e:
        traceback.print_exc()
        print(e)

