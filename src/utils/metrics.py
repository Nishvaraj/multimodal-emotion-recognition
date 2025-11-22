def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    total = y_true.size(0)
    return correct / total

def precision(y_true, y_pred, average='binary'):
    from sklearn.metrics import precision_score
    return precision_score(y_true, y_pred, average=average)

def recall(y_true, y_pred, average='binary'):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, average=average)

def f1_score(y_true, y_pred, average='binary'):
    from sklearn.metrics import f1_score
    return f1_score(y_true, y_pred, average=average)