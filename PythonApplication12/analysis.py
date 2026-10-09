
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(csv_path='data.csv'):

    df = pd.read_csv(csv_path, header=None)
    
    column_names = ['id', 'diagnosis'] + [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]
    df.columns = column_names

    print("=== 1. ОБЩАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ ===")
    print(f"Размерность: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Количество пропусков: {df.isnull().sum().sum()}")
    print("\nРаспределение целевого класса (diagnosis):")
    print(df['diagnosis'].value_counts())
    print("\nПроцентное соотношение:")
    print(df['diagnosis'].value_counts(normalize=True) * 100)

    print("\n=== 2. СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ (первые 10 признаков) ===")
    features = column_names[2:]
    stats = df[features].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']]
    stats.columns = ['Количество', 'Среднее', 'Стд. откл.', 'Мин.', 'Медиана', 'Макс.']
    print(stats.head(10))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='diagnosis', data=df, palette='Set2')
    plt.title('Распределение целевого класса (Diagnosis)', fontsize=12)
    plt.xlabel('Диагноз (B = Доброкачественная, M = Злокачественная)')
    plt.ylabel('Количество экземпляров')
    
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f'{height} ({height/len(df)*100:.1f}%)', 
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('eda_target_distribution.png', dpi=300)
    plt.close()
    print("\nСохранен график: eda_target_distribution.png")

    plt.figure(figsize=(10, 8))
    mean_features = [c for c in column_names if 'mean' in c]
    corr_matrix = df[mean_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Корреляционная матрица основных признаков (Mean Features)', fontsize=12)
    plt.tight_layout()
    plt.savefig('eda_correlation_matrix.png', dpi=300)
    plt.close()
    print("Сохранен график: eda_correlation_matrix.png")

    key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean']
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(key_features, 1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, hue='diagnosis', element='step', 
                     stat='density', common_norm=False, palette='Set1', kde=True)
        plt.title(f'Распределение: {col}')
    plt.tight_layout()
    plt.savefig('eda_key_features_distribution.png', dpi=300)
    plt.close()
    print("Сохранен график: eda_key_features_distribution.png")

if __name__ == '__main__':
    run_eda()