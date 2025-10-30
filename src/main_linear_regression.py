import pandas as pd
from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


#### stochasitc_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def stochasitc_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start

    ####### Write your code here - end
    return


#### batch_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def batch_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start

    ####### Write your code here - end
    return


def main(gd_mode, alpha):
    max_epochs = 1000

    # CSV 파일 경로 설정
    tr_file_path = '../data/train.csv'
    val_file_path = '../data/val.csv'
    test_file_path = '../data/test.csv'

    # CSV 파일을 DataFrame로 로드
    df_train = pd.read_csv(tr_file_path)
    df_val = pd.read_csv(val_file_path)
    df_test = pd.read_csv(test_file_path)
    # DataFrame의 첫 5행 출력
    print(df_train.head())
    print(df_val.head())


    tr_income = df_train['income'].values
    tr_happiness = df_train['happiness'].values

    val_income = df_val['income'].values
    val_happiness = df_val['happiness'].values

    test_income = df_test['income'].values
    test_happiness = df_test['happiness'].values

    model = LinearRegression()
    num_tr_data = len(tr_income)
    num_val_data = len(val_income)
    num_test_data = len(test_income)

    # 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_income, tr_happiness, label='real', color='blue')
    prediction = model.predict(tr_income)
    plt.plot(tr_income, prediction, label='regression', color='red')

    # 그래프 제목 및 축 레이블 추가
    plt.title('income vs. happiness (Training)')
    plt.xlabel('happiness')
    plt.ylabel('income')
    plt.legend()
    # 그리드 추가
    plt.grid(True)
    # 그래프 출력
    plt.show()

    if gd_mode == 0: # stochastic GD
        print("[[[[[ train with stochastic GD]]]]]]")
    else:
        print("[[[[[ train with batch GD]]]]]]")

    for e in range(0, max_epochs):
        if gd_mode == 0: # stochastic GD
            stochasitc_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)
        else: # batch GD
            batch_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)

        ## mse calculation
        ## Calculate MSE for training data at the end of each epoch - begin


        ## Calculate MSE for training data at the end of each epoch - end

        # print("epoch: %d, MSE: %.3f" % (e, mse))

    ## OK. Here, we finished the training - write code for evaluation - begin

    ## OK. Here, we finished the training - write code for evaluation - end

if __name__ == "__main__":
    ########## gd_mode : 0 => stochastic GD, other => batch GD
    gd_mode = 0
    ########## learning rate
    alpha = 0.01
    main(gd_mode=gd_mode, alpha=alpha)
