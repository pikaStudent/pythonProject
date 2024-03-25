import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Завантаження даних
df = pd.read_csv('data_metrics.csv')

# Встановлення порогу для класифікації
thresh = 0.5

# Прогнозування за допомогою моделі RF та LR
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')

# Підрахунок TP, FN, FP, TN для моделі RF
def find_conf_matrix_values(y_true, y_pred):
    # Обчислює значення матриці помилок
    return confusion_matrix(y_true, y_pred).ravel()

# Порівняння матриць помилок для моделі RF
def pika_accuracy_score(y_true, y_pred):
    # Обчислює точність
    TP, FP, FN, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)

# Обчислення показника accuracy_score для моделі RF
accuracy_RF = accuracy_score(df.actual_label.values, df.predicted_RF.values)

# Виведення результатів
print("Confusion Matrix for RF:")
print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))
print("Accuracy Score for RF:", accuracy_RF)

# Перевірка відповідності точності
assert np.isclose(pika_accuracy_score(df.actual_label.values, df.predicted_RF.values), accuracy_RF), 'pika_accuracy_score failed on RF'
assert np.isclose(pika_accuracy_score(df.actual_label.values, df.predicted_LR.values), accuracy_score(df.actual_label.values, df.predicted_LR.values)), 'pika_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % pika_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Accuracy LR: %.3f' % pika_accuracy_score(df.actual_label.values, df.predicted_LR.values))

# Визначення власної функції для обчислення precision_score
def pika_precision_score(y_true, y_pred):
    # Обчислення значень матриці помилок
    TN, FP, FN, TP = find_conf_matrix_values(y_true, y_pred)  # Змінено порядок
    # Обчислення precision_score
    precision = TP / (TP + FP)
    return precision

# Перевірка відповідності precision_score
assert np.isclose(pika_precision_score(df.actual_label.values, df.predicted_RF.values), precision_score(df.actual_label.values, df.predicted_RF.values)), 'pika_precision_score failed on RF'
assert np.isclose(pika_precision_score(df.actual_label.values, df.predicted_LR.values), precision_score(df.actual_label.values, df.predicted_LR.values)), 'pika_precision_score failed on LR'

print('Precision RF: %.3f' % pika_precision_score(df.actual_label.values, df.predicted_RF.values))
print('Precision LR: %.3f' % pika_precision_score(df.actual_label.values, df.predicted_LR.values))

# Визначення власної функції для обчислення F1-показника
def pika_f1_score(y_true, y_pred):
    # Обчислення значень precision і recall
    precision = pika_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Обчислення F1-показника
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

# Перевірка відповідності F1-показника
pika_f1_rf = pika_f1_score(df.actual_label.values, df.predicted_RF.values)
f1_rf = f1_score(df.actual_label.values, df.predicted_RF.values)
print('F1 RF (pika): %.3f' % pika_f1_rf)
print('F1 RF (sklearn): %.3f' % f1_rf)

pika_f1_lr = pika_f1_score(df.actual_label.values, df.predicted_LR.values)
f1_lr = f1_score(df.actual_label.values, df.predicted_LR.values)
print('F1 LR (pika): %.3f' % pika_f1_lr)
print('F1 LR (sklearn): %.3f' % f1_lr)

print('scores with threshold = 0.5')
print('Accuracy RF: %.3f' % pika_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print('Precision RF: %.3f' % pika_precision_score(df.actual_label.values, df.predicted_RF.values))
print('Recall RF: %.3f' % recall_score(df.actual_label.values, df.predicted_RF.values))
print('F1 RF: %.3f' % pika_f1_score(df.actual_label.values, df.predicted_RF.values))
print('')
print('scores with threshold = 0.25')
print('Accuracy RF: %.3f' % pika_accuracy_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('Precision RF: %.3f' % pika_precision_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('Recall RF: %.3f' % recall_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))
print('F1 RF: %.3f' % pika_f1_score(df.actual_label.values, (df.model_RF >= 0.25).astype('int').values))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# Обчислення характеристик ROC для моделей RF та LR
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# Побудова кривих ROC
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


from sklearn.metrics import roc_auc_score

# Обчислення AUC для RF та LR
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

# Виведення результатів
print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

# Візуалізація кривих ROC з позначенням AUC
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

