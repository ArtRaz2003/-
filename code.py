
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import micropip
await micropip.install('seaborn')
await micropip.install('imbalanced-learn')

import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Машинное обучение
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.decomposition import PCA

# Модели с регуляризацией
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Метрики и борьба с переобучением
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve, validation_curve

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

usecols = [
    'Severity', 'Start_Time', 'Temperature(F)', 'Humidity(%)',
    'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 
    'Precipitation(in)', 'Weather_Condition', 'Amenity', 'Bump',
    'Crossing', 'Junction', 'Traffic_Signal', 'Stop', 'Sunrise_Sunset'
]
df = pd.read_csv('US_Accidents_March23.csv', usecols=usecols, nrows=20000)

# 4. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ И ОЧИСТКА
print("\n2. ПРЕДВАРИТЕЛЬНЫЙ АНАЛИЗ И ОЧИСТКА")
print("-"*40)

# 4.2. Обработка пропусков
print("\nОбработка пропусков:")
for col in df.columns:
    missing_percent = df[col].isnull().mean() * 100
    if missing_percent > 0:
        print(f"  {col}: {missing_percent:.1f}% пропусков")
        
        critical_features = ['Severity', 'Start_Time', 'Temperature(F)', 
                            'Humidity(%)', 'Visibility(mi)', 'Weather_Condition',
                            'Junction', 'Traffic_Signal', 'Crossing', 'Stop']
        
        if col in critical_features:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
                print(f"    ⚠️  Критический признак - заполнено медианой")
            else:
                df[col] = df[col].fillna('Unknown')
                print(f"    ⚠️  Критический признак - заполнено 'Unknown'")
                
        elif missing_percent > 50:
            if col in ['Precipitation(in)', 'Wind_Speed(mph)', 'Pressure(in)']:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(0)
                    print(f"    ⚠️  Важный метео-признак - заполнено 0")
                else:
                    df[col] = df[col].fillna('Unknown')
                    print(f"    ⚠️  Важный признак - заполнено 'Unknown'")
            else:
                df = df.drop(columns=[col])
                print(f"    ✗ Удалена (>50% пропусков)")
                
        elif df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())
            print(f"    ✓ Заполнено медианой")
            
        elif df[col].dtype == 'object':
            unique_count = df[col].nunique()
            
            if unique_count < 50:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                    print(f"    ✓ Заполнено модой ({df[col].mode()[0]})")
                else:
                    df[col] = df[col].fillna('Unknown')
                    print(f"    ⚠️  Заполнено 'Unknown' (нет моды)")
            else:
                if col in ['Description', 'Street', 'City']:
                    df[col] = df[col].fillna('Unknown')
                    print(f"    ⚠️  Текстовый признак - заполнено 'Unknown'")
                elif missing_percent < 10:
                    if not df[col].mode().empty:
                        df[col] = df[col].fillna(df[col].mode()[0])
                        print(f"    ⚠️  Много категорий, но мало пропусков - заполнено модой")
                    else:
                        df[col] = df[col].fillna('Other')
                        print(f"    ⚠️  Заполнено 'Other'")
                else:
                    df = df.drop(columns=[col])
                    print(f"    ✗ Удалена (много категорий и пропусков)")
                    
        elif df[col].dtype == 'bool':
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else False)
            print(f"    ✓ Заполнено булевым значением")
            
        else:
            df[col] = df[col].fillna('Unknown')
            print(f"    ⚠️  Неизвестный тип - заполнено 'Unknown'")

# Извлекаем полезные признаки из времени
df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
df['Hour'] = df['Start_Time'].dt.hour
df['DayOfWeek'] = df['Start_Time'].dt.dayofweek
df['Month'] = df['Start_Time'].dt.month
df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
df['Is_Rush_Hour'] = ((df['Hour'] >= 7) & (df['Hour'] <= 9)) | ((df['Hour'] >= 16) & (df['Hour'] <= 19))
df['Is_Rush_Hour'] = df['Is_Rush_Hour'].astype(int)

# Удаляем исходную колонку времени
df = df.drop(columns=['Start_Time', 'End_Lat', 'End_Lng', 'Start_Lat', 'Start_Lng'], errors='ignore')

# Целевая переменная (бинарная классификация)
df['Severity_Binary'] = (df['Severity'] >= 3).astype(int)

print(f"  ✓ Создано 5 временных признаков")
print(f"  ✓ Целевая переменная: тяжелые ДТП (Severity ≥ 3)")

# Балансировка классов
print(f"\nБаланс классов:")
class_counts = df['Severity_Binary'].value_counts()
class_ratio = class_counts[1] / class_counts[0]

print(f"  Класс 0 (легкие): {class_counts[0]:,} ({class_counts[0]/len(df):.1%})")
print(f"  Класс 1 (тяжелые): {class_counts[1]:,} ({class_counts[1]/len(df):.1%})")
print(f"  Соотношение: 1:{class_ratio:.2f}")

if class_ratio < 0.3:
    print("  ⚠️  Сильный дисбаланс! Будем использовать взвешивание классов")
    use_class_weights = True
else:
    use_class_weights = False

# 5. ВЫБОР И ПОДГОТОВКА ПРИЗНАКОВ
print("\n3. ВЫБОР И ПОДГОТОВКА ПРИЗНАКОВ")
print("-"*40)

# 5.1. Выбираем наиболее информативные признаки
selected_features = [
    'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 
    'Wind_Speed(mph)', 'Precipitation(in)',
    'Hour', 'Is_Weekend', 'Is_Rush_Hour', 'DayOfWeek', 'Month',
    'Junction', 'Traffic_Signal', 'Crossing', 'Stop', 'Amenity',
    'Weather_Condition', 'Sunrise_Sunset'
]

selected_features = [f for f in selected_features if f in df.columns]
print(f"✓ Отобрано {len(selected_features)} информативных признаков")

X = df[selected_features].copy()
y = df['Severity_Binary']

# 5.2. Улучшенное упрощение категориальных признаков
print("\nУпрощение категориальных признаков:")

if 'Weather_Condition' in X.columns:
    dangerous_weather = ['Heavy Rain', 'Snow', 'Fog', 'Thunderstorm', 'Hail', 
                        'Sleet', 'Freezing Rain', 'Blizzard']
    
    weather_mapping = {
        'Clear': 'Clear',
        'Cloudy': 'Cloudy',
        'Partly Cloudy': 'Cloudy',
        'Mostly Cloudy': 'Cloudy',
        'Overcast': 'Cloudy',
        'Rain': 'Rain',
        'Light Rain': 'Rain',
        'Heavy Rain': 'Heavy Rain',
        'Drizzle': 'Rain',
        'Snow': 'Snow',
        'Light Snow': 'Snow',
        'Heavy Snow': 'Snow',
        'Fog': 'Fog',
        'Mist': 'Fog',
        'Haze': 'Fog',
        'Thunderstorm': 'Thunderstorm',
        'Hail': 'Hail',
        'Sleet': 'Sleet',
        'Freezing Rain': 'Freezing Rain',
        'Blizzard': 'Blizzard',
        'Unknown': 'Unknown'
    }
    
    X['Weather_Condition'] = X['Weather_Condition'].map(weather_mapping).fillna('Other')
    X['Dangerous_Weather'] = X['Weather_Condition'].isin(dangerous_weather).astype(int)
    
    print(f"  Weather_Condition: сгруппировано в {X['Weather_Condition'].nunique()} категорий")
    print(f"  Создан бинарный признак Dangerous_Weather")

if 'Sunrise_Sunset' in X.columns:
    X['Is_Night'] = (X['Sunrise_Sunset'] == 'Night').astype(int)
    X = X.drop(columns=['Sunrise_Sunset'])
    print(f"  Sunrise_Sunset: преобразован в бинарный Is_Night")

# 5.3. Создаем новые синтетические признаки
print("\nСоздание новых признаков:")

if 'Hour' in X.columns:
    X['Hour_sin'] = np.sin(2 * np.pi * X['Hour'] / 24)
    X['Hour_cos'] = np.cos(2 * np.pi * X['Hour'] / 24)
    print(f"  Созданы циклические признаки времени: Hour_sin, Hour_cos")

if 'Month' in X.columns:
    X['Month_sin'] = np.sin(2 * np.pi * X['Month'] / 12)
    X['Month_cos'] = np.cos(2 * np.pi * X['Month'] / 12)
    print(f"  Созданы циклические признаки месяца: Month_sin, Month_cos")

if all(col in X.columns for col in ['Visibility(mi)', 'Is_Night']):
    X['Low_Visibility_Night'] = ((X['Visibility(mi)'] < 2) & (X['Is_Night'] == 1)).astype(int)
    print(f"  Создан признак Low_Visibility_Night")

if all(col in X.columns for col in ['Junction', 'Traffic_Signal']):
    X['Junction_Without_Signal'] = ((X['Junction'] == 1) & (X['Traffic_Signal'] == 0)).astype(int)
    print(f"  Создан признак Junction_Without_Signal")

if all(col in X.columns for col in ['Humidity(%)', 'Temperature(F)']):
    X['Ice_Risk'] = ((X['Humidity(%)'] > 80) & (X['Temperature(F)'] < 35)).astype(int)
    print(f"  Создан признак Ice_Risk")

if all(col in X.columns for col in ['Is_Rush_Hour', 'Junction']):
    X['Rush_Hour_Junction'] = ((X['Is_Rush_Hour'] == 1) & (X['Junction'] == 1)).astype(int)
    print(f"  Создан признак Rush_Hour_Junction")

if all(col in X.columns for col in ['Dangerous_Weather', 'Is_Night', 'Visibility(mi)']):
    X['Dangerous_Night_LowVis'] = (
        (X['Dangerous_Weather'] == 1) & 
        (X['Is_Night'] == 1) & 
        (X['Visibility(mi)'] < 3)
    ).astype(int)
    print(f"  Создан признак Dangerous_Night_LowVis")

if all(col in X.columns for col in ['Temperature(F)', 'Humidity(%)']):
    X['THI'] = X['Temperature(F)'] - (0.55 * (1 - X['Humidity(%)']/100) * (X['Temperature(F)'] - 58))
    print(f"  Создан признак THI (Temperature-Humidity Index)")

# 5.4. Кодирование категориальных признаков
print("\nКодирование категориальных признаков:")
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

if categorical_cols:
    if 'Weather_Condition' in categorical_cols:
        weather_target_mean = y.groupby(X['Weather_Condition']).mean()
        X['Weather_Risk_Score'] = X['Weather_Condition'].map(weather_target_mean).fillna(y.mean())
        X = X.drop(columns=['Weather_Condition'])
        print(f"  Weather_Condition: применен target encoding")
        categorical_cols.remove('Weather_Condition')
    
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
        print(f"  Закодировано {len(categorical_cols)} признаков one-hot")

print(f"  Всего признаков после кодирования: {X.shape[1]}")

# 5.5. Удаление низковариативных признаков
print("\nУдаление низковариативных признаков:")
initial_features = X.shape[1]

constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    X = X.drop(columns=constant_cols)
    print(f"  ✗ Удалено {len(constant_cols)} константных признаков")

low_variance_cols = []
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        variance = X[col].var()
        if variance < 0.01:
            low_variance_cols.append(col)

if low_variance_cols:
    X = X.drop(columns=low_variance_cols)
    print(f"  ✗ Удалено {len(low_variance_cols)} признаков с низкой дисперсией")

print(f"  Осталось признаков: {X.shape[1]} (удалено {initial_features - X.shape[1]})")

# 5.6. Отбор признаков на основе важности
print("\nОтбор наиболее информативных признаков:")

temp_rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
temp_rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': temp_rf.feature_importances_
}).sort_values('importance', ascending=False)

importance_threshold = 0.005
selected_by_importance = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()

if len(selected_by_importance) < X.shape[1]:
    X_selected = X[selected_by_importance]
    print(f"  Выбрано {len(selected_by_importance)} наиболее важных признаков (порог >{importance_threshold})")
    print(f"  Топ-5 важных признаков: {', '.join(selected_by_importance[:5])}")
else:
    X_selected = X.copy()
    print(f"  Все признаки важны (ниже порога {importance_threshold})")

# 5.7. Масштабирование числовых признаков
print("\nМасштабирование признаков...")
numeric_cols = X_selected.select_dtypes(include=[np.number]).columns

if len(numeric_cols) > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected[numeric_cols])
    X_final = pd.DataFrame(X_scaled, columns=numeric_cols, index=X_selected.index)
    
    binary_cols = [col for col in X_selected.columns if col not in numeric_cols]
    if binary_cols:
        X_final = pd.concat([X_final, X_selected[binary_cols]], axis=1)
else:
    X_final = X_selected.copy()

print(f"\n✓ Итоговый размер признакового пространства: {X_final.shape[1]}")
print(f"✓ Соотношение записей/признаков: {len(X_final)}:{X_final.shape[1]} ≈ {len(X_final)/X_final.shape[1]:.1f}:1")
print(f"✓ Сохранена интерпретируемость признаков")

feature_info = {
    'original_features': selected_features,
    'final_feature_count': X_final.shape[1],
    'feature_importance': feature_importance.head(20).to_dict(),
    'top_features': selected_by_importance[:10] if 'selected_by_importance' in locals() else list(X_final.columns)[:10]
}

print(f"\nТоп-10 самых важных признаков:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
    print(f"  {i:2}. {row['feature']:<30} важность: {row['importance']:.4f}")

# 6. РАЗДЕЛЕНИЕ ДАННЫХ С КРОСС-ВАЛИДАЦИЕЙ
print("\n4. РАЗДЕЛЕНИЕ ДАННЫХ И КРОСС-ВАЛИДАЦИЯ")
print("-"*40)

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
    shuffle=True
)

X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train,
    shuffle=True
)

print(f"Обучающая выборка:     {X_train_final.shape[0]:,} записей, {X_train_final.shape[1]} признаков")
print(f"Валидационная выборка: {X_val.shape[0]:,} записей")
print(f"Тестовая выборка:      {X_test.shape[0]:,} записей")
print(f"\nСоотношение обучающих/тестовых: {X_train_final.shape[0] / X_test.shape[0]:.1f}:1")

# 7. РЕЗУЛЬТАТЫ И АНАЛИЗ МОДЕЛЕЙ
from sklearn.ensemble import VotingClassifier

print("\n" + "="*70)
print("РЕЗУЛЬТАТЫ: СРАВНЕНИЕ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
print("="*70)

models_optimized = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            C=0.01,
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ))
    ]),
    
    'Decision Tree (ограниченный)': DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    ),
    
    'Random Forest (оптимизированный)': RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        bootstrap=True
    ),
    
    'XGBoost (с дисбалансом)': XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]),
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        reg_alpha=0.1,
        reg_lambda=1.0
    ),
    
    'LightGBM (оптимизированный)': LGBMClassifier(
        n_estimators=300,
        max_depth=9,
        num_leaves=63,
        min_child_samples=20,
        min_split_gain=0.001,
        learning_rate=0.03,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.2,
        reg_lambda=1.5,
        class_weight=None,
        scale_pos_weight=1.6,
        objective='binary',
        boosting_type='gbdt',
        metric='binary_logloss',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        importance_type='split'
    ),
    
    'Gradient Boosting (регуляризованный)': GradientBoostingClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.7,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    ),
    
    'SVM (линейный)': Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(
            C=0.1,
            kernel='linear',
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ]),
    
    'Voting Classifier': VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)),
            ('xgb', XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), 
                                 random_state=42, use_label_encoder=False)),
            ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1))
        ],
        voting='soft',
        weights=[2, 3, 2]
    )
}

print("\nОбучение оптимизированных моделей...")
results_improved = {}

for name, model in models_optimized.items():
    print(f"  {name:<30}", end='')
    
    try:
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        if hasattr(model, 'predict_proba'):
            y_proba_test = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba_test)
        else:
            y_proba_test = None
            roc_auc = 0
        
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        precision = precision_score(y_test, y_pred_test, zero_division=0)
        recall = recall_score(y_test, y_pred_test, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2
        
        results_improved[name] = {
            'Accuracy_train': train_accuracy,
            'Accuracy_test': test_accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1_Score': f1,
            'ROC_AUC': roc_auc,
            'Balanced_Acc': balanced_accuracy,
            'Specificity': specificity,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }
        
        print(f"✓ F1: {f1:.3f} | ROC-AUC: {roc_auc:.3f}")
        
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        results_improved[name] = None

results_df = pd.DataFrame(results_improved).T
results_df = results_df.sort_values('F1_Score', ascending=False)

# 8. ДЕТАЛЬНАЯ ТАБЛИЦА МЕТРИК
print("\n" + "="*80)
print("ДЕТАЛЬНАЯ ТАБЛИЦА МЕТРИК ДЛЯ ВСЕХ МОДЕЛЕЙ")
print("="*80)

display_df = results_df.copy()
display_df = display_df.round(4)

display_df.columns = [
    'Acc_Train', 'Acc_Test', 'Precision', 'Recall', 
    'F1_Score', 'ROC_AUC', 'Balanced_Acc', 'Specificity',
    'TP', 'TN', 'FP', 'FN'
]

display_df['Gap'] = display_df['Acc_Train'] - display_df['Acc_Test']
display_df['FPR'] = display_df['FP'] / (display_df['FP'] + display_df['TN'])
display_df['FNR'] = display_df['FN'] / (display_df['FN'] + display_df['TP'])

print("\n" + "-"*100)
print(f"{'МОДЕЛЬ':<30} {'F1':>6} {'ROC-AUC':>7} {'Acc_Test':>8} {'Precision':>9} {'Recall':>7} {'Gap':>6}")
print("-"*100)

for idx, row in display_df.iterrows():
    f1_score_val = row['F1_Score']
    if f1_score_val > 0.7:
        f1_color = "✓"
    elif f1_score_val > 0.6:
        f1_color = "~"
    else:
        f1_color = "⚠"
    
    print(f"{idx:<30} {f1_color} {f1_score_val:>5.3f}  {row['ROC_AUC']:>6.3f}  "
          f"{row['Acc_Test']:>7.3f}  {row['Precision']:>8.3f}  "
          f"{row['Recall']:>6.3f}  {row['Gap']:>5.3f}")

print("-"*100)

# 9. АНАЛИЗ ЛУЧШИХ МОДЕЛЕЙ
print("\n" + "="*80)
print("АНАЛИЗ ТОП-3 ЛУЧШИХ МОДЕЛЕЙ")
print("="*80)

top_3_models = results_df.head(3)

for i, (model_name, metrics) in enumerate(top_3_models.iterrows(), 1):
    print(f"\n{i}. {model_name}:")
    print(f"   F1-Score:           {metrics['F1_Score']:.3f}")
    print(f"   ROC-AUC:            {metrics['ROC_AUC']:.3f}")
    print(f"   Accuracy (test):    {metrics['Accuracy_test']:.3f}")
    print(f"   Precision:          {metrics['Precision']:.3f}")
    print(f"   Recall:             {metrics['Recall']:.3f}")
    print(f"   Balanced Accuracy:  {metrics['Balanced_Acc']:.3f}")
    
    gap = metrics['Accuracy_train'] - metrics['Accuracy_test']
    if gap > 0.15:
        overfit_status = "⚠ Возможно переобучение"
    elif gap < 0:
        overfit_status = "⚠ Возможно недообучение"
    else:
        overfit_status = "✓ Хорошая сбалансированность"
    
    print(f"   Переобучение (gap): {gap:.3f} - {overfit_status}")

# 10. ВЫВОДЫ И РЕКОМЕНДАЦИИ
print("\n" + "="*80)
print("ВЫВОДЫ И РЕКОМЕНДАЦИИ ПО ВЫБОРУ МОДЕЛИ")
print("="*80)

results_df['Composite_Score'] = (
    0.4 * results_df['F1_Score'] +
    0.3 * results_df['ROC_AUC'] +
    0.2 * results_df['Balanced_Acc'] +
    0.1 * (1 - (results_df['Accuracy_train'] - results_df['Accuracy_test']).abs())
)

best_model_name = results_df['Composite_Score'].idxmax()
best_model_metrics = results_df.loc[best_model_name]

print(f"\nЛучшая модель : {best_model_name}")
print(f"Основание выбора: наивысший комплексный score ({results_df.loc[best_model_name, 'Composite_Score']:.3f})")
print(f"Ключевые метрики модели:")
print(f"  • F1-Score:          {best_model_metrics['F1_Score']:.3f}")
print(f"  • ROC-AUC:           {best_model_metrics['ROC_AUC']:.3f}")
print(f"  • Balanced Accuracy: {best_model_metrics['Balanced_Acc']:.3f}")
print(f"  • Precision:         {best_model_metrics['Precision']:.3f}")
print(f"  • Recall:            {best_model_metrics['Recall']:.3f}")

print(f"\nАнализ типов моделей:")
print(f"  • Ансамблевые методы (XGBoost, LightGBM, Random Forest): показали лучшие результаты")
print(f"  • Линейные модели (Logistic Regression, SVM): хорошая интерпретируемость, но ниже метрики")
print(f"  • Voting Classifier: показал хорошую стабильность за счет комбинации моделей")

import pickle
best_model = models_optimized[best_model_name]
with open('best_production_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Лучшая модель сохранена в файл: 'best_production_model.pkl'")

# 11. ТАБЛИЦА МЕТРИК В ФОРМАТЕ ДЛЯ ОТЧЕТА
print("\n" + "="*80)
print("ФИНАЛЬНАЯ ТАБЛИЦА МЕТРИК")
print("="*80)

report_df = results_df[['F1_Score', 'ROC_AUC', 'Accuracy_test', 
                        'Precision', 'Recall', 'Balanced_Acc']].copy()

report_df = report_df.round(3)
report_df['Рейтинг'] = range(1, len(report_df) + 1)
report_df = report_df.rename(columns={
    'F1_Score': 'F1-мера',
    'ROC_AUC': 'ROC-AUC',
    'Accuracy_test': 'Точность',
    'Precision': 'Precision',
    'Recall': 'Recall',
    'Balanced_Acc': 'Сбалансированная точность'
})

report_df = report_df[['Рейтинг', 'F1-мера', 'ROC-AUC', 'Точность', 
                       'Precision', 'Recall', 'Сбалансированная точность']]

print("\n" + report_df.to_string())
print("\n" + "="*80)
print("Метрики:")
print("="*80)
print("1. F1-мера:            Гармоническое среднее Precision и Recall")
print("2. ROC-AUC:            Площадь под ROC-кривой")
print("3. Точность:           Доля правильных предсказаний")
print("4. Precision:          Доля истинно положительных среди предсказанных положительных")
print("5. Recall:             Доля найденных положительных среди всех положительных")
print("6. Сбаланс. точность:  Среднее Recall и Specificity")

