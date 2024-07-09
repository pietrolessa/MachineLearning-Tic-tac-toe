import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
from datasetload import split_train_test

# Load and resample dataset
def load_and_resample_dataset():
    x_train, x_test, y_train, y_test = split_train_test()

    df_train = pd.DataFrame(x_train)
    df_train['class'] = y_train

    # Separar o dataset por classes
    df_game = df_train[df_train['class'] == 'game']
    df_win = df_train[df_train['class'] == 'win']
    df_lost = df_train[df_train['class'] == 'lost']
    df_tie = df_train[df_train['class'] == 'tie']

    # Aplicar oversampling nas classes minoritárias
    df_win_upsampled = resample(df_win, replace=True, n_samples=len(df_game), random_state=42)
    df_lost_upsampled = resample(df_lost, replace=True, n_samples=len(df_game), random_state=42)
    df_tie_upsampled = resample(df_tie, replace=True, n_samples=len(df_game), random_state=42)

    # Concatenar os datasets reamostrados
    df_balanced = pd.concat([df_game, df_win_upsampled, df_lost_upsampled, df_tie_upsampled])

    return df_balanced

# Carregar e reamostrar o dataset
df_balanced = load_and_resample_dataset()

# Dividir o dataset em conjuntos de treino e teste
X = df_balanced.drop('class', axis=1)
y = df_balanced['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Treinar o classificador de Árvores de Decisão
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train)

# Avaliar o modelo
y_pred = decision_tree_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
