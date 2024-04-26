def mean_error(K, Y_i, G_i, P=72):
    """
    計算平均誤差（單位：公分）。
    Args:
        K: 測試資料集的圖片數量。
        G_i: 實際氣管內管端點的 y 座標。
        Y_i: 分割氣管內管端點的 y 座標。
        P: 1 公分在圖像中轉換為 pixel 數。
    """
    # 將 y 座標轉換為公分。
    #G_cm = [y / P * 72 for y in G_i]
    #print(G_cm)
    #Y_cm = [y / P * 72 for y in Y_i]
    # 計算絕對差異。
    differences = [abs(Y - G) for Y, G in zip(Y_i, G_i)]
    #print(differences)
    # 將絕對差異加總。
    sum_differences = sum(differences)
    #print(sum_differences)
    # 計算平均誤差（單位：公分）。
    mean_error_cm = sum_differences / K
    return mean_error_cm
def accuracy_within_0_5cm(K, Y_i, G_i, P=72):
    count_within_0_5cm = sum(1 for Y, G in zip(Y_i, G_i) if abs(Y - G) <= P / 2)
    accuracy = (count_within_0_5cm / K) * 100
    return accuracy
def accuracy_within_1cm(K, Y_i, G_i, P=72):
    count_within_1cm = sum(1 for Y, G in zip(Y_i, G_i) if abs(Y - G) <= P)
    accuracy2 = (count_within_1cm / K) * 100
    return accuracy2


#test
K = 10
G_i = [200, 210, 215, 220, 225, 230, 235, 240, 245, 250]
Y_i = [198, 208, 213, 218, 227, 233, 234, 238, 243, 248]

mean_error_result = mean_error(K, Y_i, G_i)
print("平均誤差（公分）：", mean_error_result)
accuracy_within_0_5cm_result = accuracy_within_0_5cm(K, Y_i, G_i)
print("誤差在0.5cm內的準確率:", accuracy_within_0_5cm_result, "%")
accuracy_within_1cm_result = accuracy_within_1cm(K, Y_i, G_i)
print("誤差在0.5cm內的準確率:", accuracy_within_1cm_result, "%")