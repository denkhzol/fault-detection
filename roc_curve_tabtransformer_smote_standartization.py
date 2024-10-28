import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

# To load and preprocess the data from a CSV file
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(columns=['ID', 'Type', 'Name', 'LongName', 'Parent', 'Component', 'BugNo'])
    data.fillna('0.0', inplace=True)
    data = data.astype('float')
    data = data.drop(columns=['Connectors', 'InstSpec', 'LLInst', 'MsgSent', 'MsgRecv', 'MsgSelf', 'Diags', 'Dep_Out', 'Dep_In'])
    return data

# To split the data into training and testing sets
def split_data(data):
    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]
    return train_test_split(X, Y, test_size=0.25, random_state=42)

# To apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    return smote.fit_resample(X_train, y_train)

# To standardize the training and testing data
def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler

# To prepare dataframes for training and testing
def prepare_dataframes(X_train, X_test, y_train, y_test, column_names, target):
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    train = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)
    test = pd.concat([X_test, y_test.reset_index(drop=True)], axis=1)
    train.columns = column_names + target
    test.columns = column_names + target

    print("-------------- PREPARED DATASET ------------")
    print(train.shape)
    print(train.head())
    print(test.shape)
    print(test.head())
    return train, test

# To calculate various evaluation metrics
def calculate_metrics(prediction_result, y_test):
    metrics = {
        'accuracy': accuracy_score(y_test, prediction_result),
        'precision': precision_score(y_test, prediction_result, average='macro'),
        'recall': recall_score(y_test, prediction_result, average='macro'),
        'F1 score': f1_score(y_test, prediction_result, average='macro'),
        'AUC': roc_auc_score(y_test, prediction_result, average='macro')
    }
    return pd.DataFrame(metrics, index=['TabTransformer'])

# To evaluate the model by providing data and plotting the ROC curve
def evaluate_and_plot_roc(tabular_model, test_all, y_test2, filename):
    pred_df = tabular_model.predict(test_all)
    prediction_result = pred_df['prediction']
    pred_df['1.0_probability'] = pred_df['1.0_probability'].fillna(0.0)
    y_pred_proba_tabtf = pred_df['1.0_probability']
    fpr_tabtf, tpr_tabtf, _ = roc_curve(y_test2, y_pred_proba_tabtf)
    plt.plot(fpr_tabtf, tpr_tabtf, linestyle='-', label='TabTF')
    plt.legend(loc='lower right', fontsize=14)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.savefig(f"result/ROC_{filename}_TabTF.png")
    print("-------------- PILOTS SAVED ------------")
    plt.show()
    plt.clf()

# Main function to execute the entire workflow
def main():
    #file_path = 'data/AllMerged.csv'
    file_path = 'data/AllMerged.csv'
    data = load_and_preprocess_data(file_path)
    
    column_names = data.columns[:-1].tolist()
    target = data.columns[-1:].tolist()
    
    X_train, X_test, y_train, y_test = split_data(data)
    X_train, y_train = apply_smote(X_train, y_train)
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    
    train, test = prepare_dataframes(X_train, X_test, y_train, y_test, column_names, target)
    
    data_config = DataConfig(target=target, continuous_cols=column_names, categorical_cols=[])
    trainer_config = TrainerConfig(batch_size=256, max_epochs=500, early_stopping="valid_loss", 
                                   early_stopping_mode="min", early_stopping_patience=5, 
                                   checkpoints="valid_loss", load_best=True)

    optimizer_config = OptimizerConfig()
    head_config = LinearHeadConfig(layers="", dropout=0.1, initialization="uniform").__dict__
    model_config = CategoryEmbeddingModelConfig(task="classification", layers="64-32", 
                                                activation="ReLU", learning_rate=0.0005, 
                                                head="LinearHead", head_config=head_config)
    
    tabular_model = TabularModel(data_config=data_config, model_config=model_config, 
                                 optimizer_config=optimizer_config, trainer_config=trainer_config)
    tabular_model.fit(train=train)
    tabular_model.evaluate(test)
    
    pred_df = tabular_model.predict(test)
    prediction_result = pred_df['prediction']
    
    evaluation_table = calculate_metrics(prediction_result, y_test)
    print("-------------- EVALUATION ------------")
    print(evaluation_table)
    
    filenames = ['mergePromise', 'mergedBugPrediction']
    for filename in filenames:
        data2 = load_and_preprocess_data('data/' + filename + '.csv')
        print("-------------- TEST DATA ------------")
        print(data2.shape)
        print(data2.head())
        X_all = scaler.transform(data2.iloc[:, :-1])
        y_test2 = data2.iloc[:, -1]
        test_all = pd.concat([pd.DataFrame(X_all), pd.Series(y_test2).reset_index(drop=True)], axis=1)
        test_all.columns = column_names + target

        print("-------------- TEST DATA AFTER STANDARDIZATION ------------")
        print(test_all.shape)
        print(test_all.head())

        tabular_model.evaluate(test_all)
        evaluate_and_plot_roc(tabular_model, test_all, y_test2, filename)

if __name__ == "__main__":
    main()

    print("-------------- FINISHED SUCCESSFULLY ------------")
