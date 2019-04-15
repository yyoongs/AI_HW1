import numpy as np

# 2019-04-15 조용성
# 최종 제출
class singleLayer:
    def __init__(self, W, Bias):  # 제공. 호출 시 작동하는 생성자
        self.W = W
        self.B = Bias

    def SetParams(self, W_params, Bias_params):  # 제공. W와 Bias를 바꾸고 싶을 때 쓰는 함수
        self.W = W_params
        self.B = Bias_params

    def ScoreFunction(self, X):  # Score값 계산 -> 직접작성
        # 3.2
        # X의 shape는 (60000,784)
        # W의 shape는 (784,10)
        # ScoreMatrix는 (60000,10)이 되어야함
        ScoreMatrix = np.dot(X, self.W) + self.B
        # shape = (60000,784) @ (784,10) + (10)
        return ScoreMatrix

    def Softmax(self, ScoreMatrix):  # 제공.
        if ScoreMatrix.ndim == 2:
            temp = ScoreMatrix.T
            temp = temp - np.max(temp, axis=0)
            y_predict = np.exp(temp) / np.sum(np.exp(temp), axis=0)
            return y_predict.T
        temp = ScoreMatrix - np.max(ScoreMatrix, axis=0)
        expX = np.exp(temp)
        y_predict = expX / np.sum(expX)
        return y_predict  # 모든 클래스의 합이1인 exp까지 계산

    def LossFunction(self, y_predict, Y):  # Loss Function을 구하십시오 -> 직접 작성
        # 3.3
        # (Softmax의 결과 값 : y_predict, 정답값 : Y)
        epsilon = 1e-7
        # log0이 나올 때의 오류를 제거하기 위해 충분히 작은 양의 실수 epsilon 변수 설정
        # y_predict의 차원의 크기가 2일 때
        if y_predict.ndim == 2:
            test = np.sum(y_predict * Y, axis=1)
            temp = - np.log(test + epsilon)
            # shape가 (60000,10)인 y_predict와 (60000,10)의 정답값 레이블 Y와 곱하여 가로로 10개의 값을 sum 적용
            # 가로로 10개중 의미있는 값은 하나이므로 test = np.max(y_predict * Y, axis=1)와 같은 값을 가짐
            # 나온 결과 값에 epsilon을 더하여 log를 취함
            # 결과는 각 이미지당 -log(p(x))*(q(x)) 값을 가지는 60000개의 loss 값
            loss = temp.sum() / temp.shape[0]
            return loss
        # 60000개의 loss값을 다 더해서 60000으로 나눔
        # 모든 loss의 평균치를 return

        # 차원의 크기가 2가 아닐때 ( 차원의 크기가 1일 때 )
        test_n = y_predict * Y
        temp_n = -np.log(test_n + epsilon)
        loss_n = temp_n.sum() / temp_n.shape
        # 위와 같은 원리로 loss_n 값 계산
        return loss_n

    def Forward(self, X, Y):  # ScoreFunction과 Softmax, LossFunction를 적절히 활용해 y_predict 와 loss를 리턴시키는 함수. -> 직접 작성
        # 3.4
        y_score = self.ScoreFunction(X)
        # score값을 계산하여 y_score에 넣음
        y_predict = self.Softmax(y_score)
        # 위에서 계산한 y_score값을 가지고 softmax함수를 적용하여 y_predict 결과 값을 생성
        loss = self.LossFunction(y_predict, Y)
        # 위에서 계산한 y_predict값을 가지고 정답값 레이블인 Y와 함께 loss값을 생성
        return y_predict, loss

    def delta_Loss_Scorefunction(self, y_predict, Y):  # 제공.dL/dScoreFunction
        delta_Score = y_predict - Y
        return delta_Score
    # delta_Score의 shape = (60000,10)

    def delta_Score_weight(self, delta_Score, X):  # 제공. dScoreFunction / dw .
        delta_W = np.dot(X.T, delta_Score) / X[0].shape
        return delta_W
    # delta_W의 shape = (784,10)

    def delta_Score_bias(self, delta_Score, X):  # 제공. dScoreFunction / db .
        delta_B = np.sum(delta_Score) / X[0].shape
        return delta_B
    # delta_B의 shape = (1)

    # delta 함수를 적절히 써서 delta_w, delta_b 를 return 하십시오.
    def BackPropagation(self, X, y_predict, Y):
        # 3.5
        delta_score = self.delta_Loss_Scorefunction(y_predict, Y)
        # 주어진 loss_scorefunction으로 dL/dScoreFunction을 구함
        delta_W = self.delta_Score_weight(delta_score, X)
        # chain-rule을 적용하기위해 위에서 구한 dL/dScoreFunction을 이용하여 delta_Score_weight 함수에 넣어 dL/dw 값을 계산
        delta_B = self.delta_Score_bias(delta_score, X)
        # 마찬가지로 위에서 구한 dL/dScoreFunction을 이용하여 delta_Score_bias 함수에 넣어 dL/db 값을 계산
        return delta_W, delta_B

    # 정확도를 체크하는 Accuracy 제공
    def Accuracy(self, X, Y):
        y_score = self.ScoreFunction(X)
        y_score_argmax = np.argmax(y_score, axis=1)
        if Y.ndim != 1: Y = np.argmax(Y, axis=1)
        accuracy = 100 * np.sum(y_score_argmax == Y) / X.shape[0]
        return accuracy

    # Forward와 BackPropagationAndTraining, Accuracy를 사용하여서 Training을 epoch만큼 시키고, 10번째 트레이닝마다
    # Training Set의 Accuracy 값과 Test Set의 Accuracy를 print 하십시오

    def Optimization(self, X_train, Y_train, X_test, Y_test, learning_rate=0.01, epoch=100):
        for i in range(epoch):
            # 3.6
            y_predict, loss = self.Forward(X_train, Y_train)
            # 위에서 구현한 Forward 함수를 사용하여 y_predict와 loss 값을 리턴
            delta_W, delta_B = self.BackPropagation(X_train, y_predict, Y_train)
            # Forward 함수로 구한 y_predict를 사용해 위에서 구현한 BackPropagation 함수에 넣어 delta_W와 delta_B 값을 계산
            self.SetParams(self.W - delta_W * learning_rate, self.B - delta_B * learning_rate)
            # 위에서 구한 delta_W값에 learning_rate를 곱해 각각 W와 B에 SetParams함수를 이용해 값에 변화를 준다.
            # 함수 작성
            if i % 10 == 0:
                # 3.6 Accuracy 함수 사용
                print(i, "번째 트레이닝")
                print('현재 Loss(Cost)의 값 : ', loss)
                train_acc = self.Accuracy(X_train, Y_train)
                # 정확도 체크 함수 Accuracy를 사용하여 training set의 정확도를 계산
                print("Train Set의 Accuracy의 값 : ", train_acc)
                test_acc = self.Accuracy(X_test, Y_test)
                # 정확도 체크 함수 Accuracy를 사용하여 test set의 정확도를 계산
                print("Test Set의 Accuracy의 값 :", test_acc)


    def Numerical_Gradient(self, X, Y, h=0.000001):
        delta_W = np.zeros(([784,10]))
        # dW값을 넣을 넘파이 어레이 shape 784,10을 만들고 모든 값은 0으로 초기화
        param_h = np.zeros([784, 10])
        # numerical 값을 더하기 위해 W와 같은 shape를 가진 param_h 배열을 모든 원소 값 0으로 초기화
        y_predict, loss = self.Forward(X, Y)
        # x_train과 y_train을 Forward 하여 최초 y_predict와 loss값을 구함
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                param_h[i][j] = h
                # param_h 배열의 [i][j]원소를 h로 설정
                # 나머지 값은 0
                self.SetParams(self.W + h, self.B)
                # self.W에 h를 더해줌
                y_predict_2, loss_h = self.Forward(X, Y)
                # 변화된 self.W를 가지고 다시 Forward 진행
                delta_W[i][j] = (loss_h - loss) / h
                # delta_W[i][j]는 변화된 W로 구한 loss값에 최초 loss값을 빼고 h로 나눈 값이다.
                param_h[i][j] = 0
                # 바꿨던 param_h[i][j]값을 다시 0으로 초기화
        # 위와 같은 방법을 반복하여 delta_W의 각각의 원소에 대한 numerical gradient를 계산하여 delta_W에 저장
        return delta_W