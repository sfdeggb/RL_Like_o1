#stage1 :训练模型初始化

def train_initialzation(base_model,data,epchos):
    model  =copy(base_model)
    for i in range(epchos):
        for (X,Y_star) in data:
            Y1 = base_model.predict(X)
            Y2 =model.predict(X)
            reward = reward_function(Y2,Y_star)
            model.update(X,Y_star,reward)
            loss = -reward+beta2 *kl_divergence(Y1,Y2)
            model.update(loss)


def reward_function(Y1,Y2):
    return 1 if Y2 >= Y1 else 0
