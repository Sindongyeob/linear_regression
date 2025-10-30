import pandas as pd
from linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


#### stochastic_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def stochastic_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start
    for x, y in zip(x_list, y_list):
        grad = model.gradient_of_SE(x, y)
        model.update_params(grad[0], grad[1], alpha)

    ####### Write your code here - end
    return


#### batch_gd()
#### x_list : input, y_list : ground truth, model : linear regression model, alpha : learning rate
def batch_gd(x_list, y_list, model, alpha):
    ####### Write your code here - start
    n = len(x_list)
    grad_w_sum = 0.0
    grad_b_sum = 0.0

    for x, y in zip(x_list, y_list):
        grad = model.gradient_of_SE(x, y)
        grad_w_sum += grad[0]
        grad_b_sum += grad[1]

    grad_w_avg = grad_w_sum/n
    grad_b_avg = grad_b_sum/n

    model.update_params(grad_w_avg, grad_b_avg, alpha)
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

    #w, b 초기값
    init_w = model.w
    init_b = model.b
    print("\n===== Initial Model Parameters =====")
    print(f"w: {init_w:.6f}, b: {init_b:.6f}")

    # 학습 전 그래프 생성
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_income, tr_happiness, label='real', color='blue')
    prediction = model.predict(tr_income)
    plt.plot(tr_income, prediction, label='regression', color='red')

    # 학습 전 그래프 제목 및 축 레이블 추가
    plt.title('income vs. happiness (before Training)')
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
            stochastic_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)
        else: # batch GD
            batch_gd(x_list=tr_income, y_list=tr_happiness, model=model, alpha=alpha)

        ## mse calculation
        ## Calculate MSE for training data at the end of each epoch - begin
        mse = 0.0
        for x, y in zip(tr_income, tr_happiness):
            se = model.SE(x, y)
            mse += se
        mse = mse / num_tr_data
        ## Calculate MSE for training data at the end of each epoch - end

        print("epoch: %d, MSE: %.3f" % (e, mse))

    ## OK. Here, we finished the training - write code for evaluation - begin
    # 학습 후 training data 및 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(tr_income, tr_happiness, label='Training Data', color='blue')
    prediction = model.predict(tr_income)
    plt.plot(tr_income, prediction, label='Trained Model', color='red')
    plt.title('Income vs. Happiness (After Training - Training Data)')
    plt.xlabel('Income')
    plt.ylabel('Happiness')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Validation data 및 학습된 모델 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(val_income, val_happiness, label='Validation Data', color='blue')
    val_prediction = model.predict(val_income)
    plt.plot(val_income, val_prediction, label='Trained Model', color='red')
    plt.title('Income vs. Happiness (Validation Data)')
    plt.xlabel('Income')
    plt.ylabel('Happiness')
    plt.legend()
    plt.grid(True)
    plt.show()

    #Test data 및 학습된 모델 그래프 출력
    plt.figure(figsize=(8, 6))
    plt.scatter(test_income, test_happiness, label='Test Data', color='blue')
    test_prediction = model.predict(test_income)
    plt.plot(test_income, test_prediction, label='Trained Model', color='red')
    plt.title('Income vs. Happiness (Test Data)')
    plt.xlabel('Income')
    plt.ylabel('Happiness')
    plt.legend()
    plt.grid(True)
    plt.show()

    valid_mse = 0.0
    for x, y in zip(val_income, val_happiness):
        se = model.SE(x, y)
        valid_mse += se
    valid_mse = valid_mse / num_val_data

    # Test MSE 계산
    test_mse = 0.0
    for x, y in zip(test_income, test_happiness):
        se = model.SE(x, y)
        test_mse += se
    test_mse = test_mse / num_test_data

    print(f"\n||||||||           Final Results            ||||||||")
    print(f"Initial Model Parameters - w: {init_w:.6f}, b: {init_b:.6f}")
    print(f"Final Model Parameters   - w: {model.w:.6f}, b: {model.b:.6f}")
    print(f"Validation MSE: {valid_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    ## OK. Here, we finished the training - write code for evaluation - end

if __name__ == "__main__":
    ########## gd_mode : 0 => stochastic GD, other => batch GD
    gd_mode = 0
    ########## learning rate
    alpha = 0.01
    main(gd_mode=gd_mode, alpha=alpha)
