def reinforce_train(model,data,epchos):
    for epcho in range(epchos):
        for (X,Y_star) in data:
            Y1 = model.predict(X)
            reward = reward_function(Y1,Y_star)
            Y2 = model.predict(X,Y1)
            reward1 = reward_function(Y1,Y_star)
            reward2 = reward_function(Y2,Y_star)
            bonus = kappa * (reward2-reward1)
            loss = -reward2+beta2 *kl_divergence(Y1,Y2)+bonus
            model.update(loss)
    return model 

def reward_function(Y1,Y_star):
    return 1 if Y1 >= Y_star else 0

def kl_divergence(Y1,Y2):
            reward2 = reward_function(Y2,Y_star)

