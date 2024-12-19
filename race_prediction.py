import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

directory_path=r'C:\Users\surej\OneDrive\Desktop\project\horse\Horse'
csv_files = [f for f in os.listdir(directory_path) if f.endswith(".csv")]

all_dfs_horse=[]
all_dfs_races=[]

for file in csv_files:
    file_path=os.path.join(directory_path,file)
    if 'horses' in file_path:
        df=pd.read_csv(file_path)
        all_dfs_horse.append(df)
    elif 'races' in file_path:
        df=pd.read_csv(file_path)
        all_dfs_races.append(df)

horses_dfs = pd.concat(all_dfs_horse, ignore_index=True)
races_dfs = pd.concat(all_dfs_races, ignore_index=True)

directory_path=r'C:\Users\surej\OneDrive\Desktop\project\horse'

output_file = os.path.join(directory_path, 'horses_combined.csv')
horses_dfs.to_csv(output_file, index=False)

output_file = os.path.join(directory_path, 'races_combined.csv')
races_dfs.to_csv(output_file, index=False)

       
df=pd.read_csv(r"C:\Users\surej\OneDrive\Desktop\project\horse\horses_combined.csv")
print(len(df))
df['TR'] = df.groupby('horseName')['TR'].transform(lambda x: x.fillna(x.median()))
df['RPR'] = df.groupby('horseName')['RPR'].transform(lambda x: x.fillna(x.median()))
mis=df.isna().sum()
print(mis)
threshold=0.2*len(df)
columns_to_drop=mis[mis>threshold].index
print(columns_to_drop)
df_clean=df.drop(columns=columns_to_drop,axis=1)
df_clean.dropna(inplace=True)
print(len(df_clean))

new_folder=r'C:\Users\surej\OneDrive\Desktop\project\horse\cleaned_datasets'
original_filename = 'horses_cleaned.csv'
new_file_path = os.path.join(new_folder, original_filename)
df_clean.to_csv(new_file_path,index=False)

df=pd.read_csv(r"C:\Users\surej\OneDrive\Desktop\project\horse\races_combined.csv")
print(len(df))
mis=df.isna().sum()
print(mis)
threshold=0.3*len(df)
columns_to_drop=mis[mis>threshold].index
print(columns_to_drop)
df_clean=df.drop(columns=columns_to_drop,axis=1)
df_clean.dropna(inplace=True)
print(len(df_clean))

new_folder=r'C:\Users\surej\OneDrive\Desktop\project\horse\cleaned_datasets'
original_filename = 'races_cleaned.csv'
new_file_path = os.path.join(new_folder, original_filename)
df_clean.to_csv(new_file_path,index=False)

df=pd.read_csv(r"C:\Users\surej\OneDrive\Desktop\project\horse\cleaned_datasets\horses_cleaned.csv")

df.isna().sum()

horse_df = pd.read_csv(r'C:\Users\surej\OneDrive\Desktop\project\horse\cleaned_datasets\horses_cleaned.csv')
race_df = pd.read_csv(r'C:\Users\surej\OneDrive\Desktop\project\horse\cleaned_datasets\races_cleaned.csv')

combined_df = pd.merge(horse_df, race_df, on='rid', how='inner')

combined_df['speed_weight_ratio'] = combined_df['TR'] / combined_df['weight']
combined_df['age_weight_ratio'] = combined_df['age'] / combined_df['weight']
combined_df['winning_percentage'] = combined_df['res_win'] / combined_df['runners']

horse_perf = combined_df.groupby('horseName').agg({
    'res_win': ['sum', 'mean'],
    'res_place': ['sum', 'mean'],
    'position': 'mean',
    'TR': 'mean',
    'decimalPrice': 'mean'
}).reset_index()

horse_perf.columns = ['horseName', 'total_wins', 'win_rate', 'total_places', 
                      'place_rate', 'avg_position', 'avg_topspeed', 'avg_price']

combined_df = pd.merge(combined_df, horse_perf, on='horseName', how='left')

print(combined_df.describe())

plt.figure(figsize=(12, 8))
sns.histplot(combined_df['TR'], kde=True, bins=30, color='blue')
plt.title('Distribution of Topspeed (TR)')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(combined_df['weight'], kde=True, bins=30, color='green')
plt.title('Distribution of Horse Weight')
plt.show()

plt.figure(figsize=(12, 10))
corr = combined_df[['TR', 'weight', 'decimalPrice', 'position', 'total_wins', 'win_rate']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

combined_df_refined=combined_df.drop(columns=[
    'rid', 'horseName', 'trainerName', 'jockeyName',  
    'father', 'mother', 'gfather', 'res_place',                          
    'course', 'time', 'date', 'title',               
    'weightSt', 'weightLb'                           
])

categorical_columns = combined_df_refined.select_dtypes(include=['object']).columns
numeric_columns = combined_df_refined.select_dtypes(include=['int64', 'float64']).columns

label_encoder = LabelEncoder()
for col in categorical_columns:
    
    combined_df_refined[col] = combined_df_refined[col].astype(str)
    combined_df_refined[col] = label_encoder.fit_transform(combined_df_refined[col])
    print(f"Encoded {col}")

X = combined_df_refined.drop('res_win',axis=1)
y = combined_df_refined['res_win']

X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search_rf.fit(X_resampled, y_resampled)          

best_rf = grid_search_rf.best_estimator_
print("Best Random Forest Parameters:", grid_search_rf.best_params_)

# Evaluate Model
y_pred = best_rf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

