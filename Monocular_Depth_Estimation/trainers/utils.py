import numpy as np

def interval_score(y_true, pred_lower, pred_upper, alpha=0.05):
    width = pred_upper - pred_lower

    # Penalties
    below = y_true < pred_lower
    above = y_true > pred_upper

    penalty = (
        (2 / alpha) * (pred_lower - y_true) * below +
        (2 / alpha) * (y_true - pred_upper) * above
    )

    score = width + penalty
    score = score.view(score.shape[0], -1).mean(dim=1)
    width = width.view(score.shape[0], -1).mean(dim=1)

    return score,width

def picp(y_true, pred_lower, pred_upper):
    within_interval = ((y_true >= pred_lower) & (y_true <= pred_upper)).float()
    picp = within_interval.view(within_interval.shape[0], -1).mean(dim=1)
    return picp

def picp_up(y_true, pred, pred_lower, pred_upper):
    picp_up=[]
    for i in range(y_true.shape[0]):
        within_interval_up = (y_true[i][pred[i] < y_true[i]] <= pred_upper[i][pred[i] < y_true[i]]).cpu().numpy().astype(float)
        picp_up.append(np.mean(within_interval_up))
    return picp_up

def picp_down(y_true, pred, pred_lower, pred_upper):
    picp_down=[]
    for i in range(y_true.shape[0]):
        within_interval_down = (y_true[i][pred[i] > y_true[i]] >= pred_lower[i][pred[i] > y_true[i]]).cpu().numpy().astype(float)
        picp_down.append(np.mean(within_interval_down))
    return picp_down