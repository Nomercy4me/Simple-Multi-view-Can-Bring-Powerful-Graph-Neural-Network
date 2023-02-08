# -*- coding:UTF-8 -*-
from utils import process
dataset = "doubanmovie"
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
node_num = adj.shape[0]
y = y_train + y_val + y_test
print(y)

dic = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0,10:0}
cnt2 = 0
# 统计相连的节点的label一致的情况
for i in range(node_num):
    for j in range(i+1, node_num):
        if adj[i,j] ==0:
            continue
        # 仅关注连边
        print(y[i])
        print(y[j])
        cnt = 0.0
        cnt_s = 0.0
        for k in range(28):
            if y[i,k] ==1 and y[j,k] ==1:
                cnt += 1
            if y[i,k] ==1 or y[j,k] ==1:
                cnt_s += 1
        idx = int((cnt/cnt_s)*10)
        print(cnt,cnt_s,idx)
        dic[idx] += 1
        # if(cnt/cnt_s*10):
        #     dic[-1] += 1
        # else:
        #     dic[cnt] += 1
        print("********************")
        if((y[i]== y[j]).all()):
            print(y[i],y[j],"True")
            cnt2+=1

print(dic)
print(cnt2)

