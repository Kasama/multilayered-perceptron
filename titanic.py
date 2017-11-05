import os
import sys
import mlp
import pickle
import re as regex
import numpy as np
import pandas as pd


def load_dataset(train_file, predict_file=False):

    # Prepare dataset
    raw_dataset = pd.read_csv(train_file)

    # Use these columns to train the model
    columns = ['pclass', 'title', 'sex',
               'age', 'family_size', 'alone',
               'fare', 'embarked']
    if not predict_file:
        columns = ['survived'] + columns

    dataset_length = range(raw_dataset.index.size)
    dataset = pd.DataFrame(index=dataset_length, columns=columns)

    # create expected answer column
    if not predict_file:
        dataset['survived'] = raw_dataset['Survived']

# pclass: numbers 1, 2 or 3. We divide by 3 after subtracting 1 to get a
#         dataset between 0 and 1
    dataset['pclass'] = (raw_dataset['Pclass'] - 1) / 3

# title: Based on the name of the person, get him/her social title
#        - Mlle, Ms, Miss are considered the same title
#        - Mrs, Mme are considered the same title
#        - All rare titles are considered equal
    def get_title(name):
        match = regex.search(' (\w+)\.', name)
        if match:
            return match.group(1)
        return "None."
    dataset['title'] = raw_dataset['Name'].apply(get_title)
    dataset['title'] = dataset['title'].replace([
        'Capt', 'Col', 'Countess', 'Don', 'Dona'
        'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir',
        ], 'Especial')
    dataset['title'] = dataset['title'].replace(['Mlle', 'Ms'], 'Miss')
    dataset['title'] = dataset['title'].replace(['Mme'], 'Mrs')
    dataset['title'] = dataset['title'].map({
        'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3, 'Especial': 4
        }) / 5
    dataset['title'] = dataset['title'].fillna(4)
# sex: 0 for female and 1 for male
    dataset['sex'] = raw_dataset['Sex'].map({
        'female': 0, 'male': 1
        }).astype(float)

# age: group age in 5 categories:
#      - less than 18
#      - between 18 and 32
#      - between 32 and 50
#      - between 50 and 64
#      - older than 64
    dataset['age'] = raw_dataset['Age']
    age_mean = dataset['age'].mean()
    dataset['age'] = dataset['age'].fillna(age_mean)
    dataset.loc[dataset['age'] <= 18, 'age'] = 0
    dataset.loc[(dataset['age'] > 18) & (dataset['age'] <= 32), 'age'] = 1
    dataset.loc[(dataset['age'] > 32) & (dataset['age'] <= 50), 'age'] = 2
    dataset.loc[(dataset['age'] > 50) & (dataset['age'] <= 64), 'age'] = 3
    dataset.loc[dataset['age'] > 64, 'age'] = 4
    dataset['age'] = dataset['age'].astype(float) / 5

# family_size: group number of family members in 4 categories
#              - alone = 1
#              - small <= 4
#              - medium <= 8
#              - big > 8
    dataset['family_size'] = \
        (raw_dataset['SibSp'] + raw_dataset['Parch'] + 1).astype(float)
    dataset.loc[dataset['family_size'] == 1, 'family_size'] = 0
    dataset.loc[
            (dataset['family_size'] > 1) & (dataset['family_size'] <= 4),
            'family_size'
            ] = 1
    dataset.loc[
            (dataset['family_size'] > 4) & (dataset['family_size'] <= 8),
            'family_size'
            ] = 2
    dataset.loc[dataset['family_size'] > 8, 'family_size'] = 3
    dataset['family_size'] = dataset['family_size'].astype(float) / 4

# alone: 1 if travelling alone, 0 otherwise
    dataset['alone'] = \
        dataset['family_size'].apply(lambda size: 1 if size == 0 else 0)

# fare: grouped in 4 categories
#       - cheap <= 8
#       - normal <= 14
#       - expensive <= 31
#       - outrageous > 31
    dataset['fare'] = \
        raw_dataset['Fare'].fillna(raw_dataset['Fare'].median()).astype(float)
    dataset.loc[dataset['fare'] <= 8, 'fare'] = 0
    dataset.loc[(dataset['fare'] > 8) & (dataset['fare'] <= 14), 'fare'] = 1
    dataset.loc[(dataset['fare'] > 14) & (dataset['fare'] <= 31), 'fare'] = 2
    dataset.loc[dataset['fare'] > 31, 'fare'] = 3
    dataset['fare'] = dataset['fare'] / 4

# embarked
    dataset['embarked'] = raw_dataset['Embarked'].fillna('S')
    dataset['embarked'] = dataset['embarked'].map({'S': 0, 'C': 1, 'Q': 3}) / 3

    return dataset.as_matrix(columns)


# train the model using train_file, eta and threshold.
# recover from saved_model and save results to saved_model {
def train_model(train_file, hidden_neurons, mode, eta, threshold, saved_model):
    dataset = load_dataset(train_file, False)
    print('loaded test file!')
    X = dataset[:, 1:len(dataset[0])]
    Y = dataset[:, 0]

    if mode == 'recover' and os.path.isfile(saved_model):
        model = pickle.load(open(saved_model, 'rb'))
    else:
        model = mlp.MLP(
                input_layer_neurons=X.shape[1],
                hidden_layer_neurons=hidden_neurons,
                output_layer_neurons=1
                )
        if not os.path.isdir('trained'):
            os.mkdir('trained')

    print('training network with', hidden_neurons, 'hidden neurons:')
    sys.stdout.flush()
    model.learn(X, Y, eta, threshold, saved_model)
    if (mode != 'predict'):
        print('trained!')
        sys.stdout.flush()
# == end train_model }


# test the model against test file and check answers to get score
def test_model(model, test_file):  # {
    dataset = load_dataset(test_file)
    print('loaded test file!')
    sys.stdout.flush()
    X = dataset[:, 1:len(dataset[0])]
    Y = dataset[:, 0]

    tries = len(X)
    success = 0
    for test in range(tries):
        x = X[test]
        y = int(Y[test])
        f_h, df_h, f_o, df_o = model.feed_forward(x)
        if(np.round(f_o) == y):
            success += 1
    print('got ', success, '/', tries, ': ', success*100/tries)
    sys.stdout.flush()
# == end test_model }


# test the model against prediction file and output answers
def predict_model(model, predict_file):  # {
    dataset = load_dataset(predict_file, True)
    X = dataset
    print('PassengerId,Survived')
    for test in range(len(X)):
        x = X[test]
        f_h, df_h, f_o, df_o = model.feed_forward(x)
        print(
                str(test + 892) +
                ',' +
                str(int(round(f_o[0])))
                )
        sys.stdout.flush()
# ==end predict_model }


""" {
Prepare the  dataset  and  run  MLP to train or predict results for the titanic
test https://www.kaggle.com/c/titanic.
--- arguments ---
file - name of the dataset to use. Must be a valid directory under ./data
mode - must be one of 'train', 'test' or 'predict'.
     - 'train': force  training to start, even if there  is a partially trained
                file is present.
     - 'test' : if there is a already trained  saved model. Use it  against the
                test file and compare results. Otherwise  train the network and
                then test it.
     - 'predict' : if there is a  already trained saved  model, use  it against
                   the prediction file  to generate the output  to be submitted
                   to kaggle at https://www.kaggle.com/c/titanic.
hidden_neurons - the number of hidden layer neurons to use in the model.
eta - the step to be used in the backpropagation algorithm while training.
threshold - the maximum allowed error for the model while training.
} """


def titanic(
        file, mode='train',
        hidden_neurons=10, eta=0.4, threshold=1e-1
        ):  # {
    train_file = 'data/' + file + '/train.csv'
    test_file = 'data/' + file + '/test.csv'
    predict_file = 'data/' + file + '/predict.csv'
    saved_model = 'trained/' + file + '-' + str(hidden_neurons) + '.mlp'

    if (mode == 'test' or mode == 'predict') and os.path.isfile(saved_model):
        model = pickle.load(open(saved_model, 'rb'))
    else:
        model = train_model(
                train_file, hidden_neurons, mode,
                eta, threshold,
                saved_model
                )

    if (mode == 'test'):
        test_model(model, test_file)
    elif (mode == 'predict'):
        predict_model(model, predict_file)
# ==end digit_recognize }


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        titanic(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        titanic(sys.argv[1], 'test')
