**utf-8**

'''
Contents
1) Prepare Problem
1.1) load libraries
1.2) load and explore the shape of the dataset

2) Summarize Data
2.1) Descriptive statistics
2.2) Visualization

3) Prepare Data
3.1) Cleaning
3.2) split out train/test dataset

4) Evaluate Algorithms
4.1) Algorithms

5) Improve Accuracy
5.1) Grid Search

6) Performance of the best algorithms
6.1) check the performance
6.2) futher process

7) Fianlize Model 
7.1) create fianal model
7.2) predictions on test dataset

'''

#1) Prepare Problem
# 1.1) load libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#데이터 전처리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import Counter

#데이터 모델
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#데이터 모델 성능 평가
from sklearn.metrics import  make_scorer
from sklearn.metrics import log_loss

plant_train = pd.read_csv('plant_train.csv')
plant_test = pd.read_csv('plant_test.csv')

# 1.2) load and explore the shape of the dataset
plant_train.head(3)
plant_test.head(3)

plant_train = plant_train.drop(['id'], axis=1)
plant_test = plant_test.drop(['id'], axis=1)

plant_train.info()

#2) Summarize Data

#2.1) Descriptive statistics

plant_train.describe()

#type -> target value
#fiberid -> categorical value
#psfMag_u의 mean: -6.750146e+00

plant_train.type.value_counts()
# unbalanced taget value

# 2.2) Visualization

# fiberID의 categorical value의 빈도수 확인
plant_train_fiberid_count = plant_train['fiberID'].value_counts()
sns.set(style="whitegrid")
sns.barplot(plant_train_fiberid_count.index[:11], plant_train_fiberid_count.values[:11], alpha=0.9)
plt.title('Freqency Distribution of fiberID')
plt.ylabel('Number of Occurences')
plt.xlabel('fiberID')
plt.show()

features = plant_train.drop(['type', 'fiberID'], axis=1)
features = features.columns.tolist()

for feature in features:
    skew = plant_train[feature].skew()
    sns.distplot(plant_train[feature], kde=False, label='Skew= %.3f' % (skew), bins=30)
    plt.legend(loc='best')
    plt.show()
# normally distrubited x


corr = plant_train[features].corr(method='pearson')
plt.figure(figsize=(20,20))
sns.heatmap(data=corr,annot=True)
plt.show()
#psfMag_u & fiberMag_u와 동일


#3) Prepare Data
# 3.1) Cleaning

# 이상치 검출
def outlier_detect(df):
    outlier_indices = []
    # iterate over features(columns)
    for feature in features:
        # 1st quartile (25%)
        Q1 = np.percentile(plant_train[feature], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(plant_train[feature], 75)
        # Interquartile rrange (IQR)
        IQR = Q3 - Q1
        # outlier step
        outlier_step = 1.5 * IQR

        # outlier가 존재하는 feature 추출
        outlier_list_feature = plant_train[(plant_train[feature] < Q1 - outlier_step) |
                                           (plant_train[feature] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_feature)

    # 10개 이상 이상치가 존재하는 관측치 검출
    outlier_indices = Counter(outlier_indices)
    # itmes는 key value로 묶어서 list return
    multiple_outliers = [k for k, v in outlier_indices.items() if v > 5]

    return multiple_outliers


print('데이터 셋에서 5개이상의 이상치가 존재하는 관측치는 %d개이다.' % (len(outlier_detect(plant_train[features]))))

# 이상치 검출된 관측치 제거
outlier_indices = outlier_detect(plant_train[features])
plant_train = plant_train.drop(outlier_indices).reset_index(drop=True)
print(plant_train.shape)
#199991에서 199635로 줄어들음

# psfMag_u 평균값과 상관분석의 결과로 보아 해당 칼럼 삭제
plant_train = plant_train.drop(['psfMag_u','fiberID'], axis=1)

# type변수 type을 object -> int변환
encoder = LabelEncoder()
plant_train['type'] = encoder.fit_transform(plant_train['type'])
plant_train.type.value_counts()

# 3.2) split out train/test dataset

X = plant_train.drop(['type'], axis=1)
y = plant_train['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=777)

print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)

# 4) Evaluate Algorithms
# 4.1) Algorithms

# randomforest
random_df = RandomForestClassifier(class_weight='balanced',random_state=777)
random_df.fit(X_train,y_train)
random_df_pred = random_df.predict_proba(X_test)
random_df_pred_log = log_loss(y_test, random_df_pred)
print('기본 randomforest logloss 값:', random_df_pred_log)
# 기본 randomforest logloss 값: 1.3466678834352566

# Bagging+DecisionTree
from sklearn.ensemble import BaggingClassifier
bag_decision_df = BaggingClassifier(DecisionTreeClassifier())
bag_decision_df.fit(X_train, y_train)
bag_df_pred = bag_decision_df.predict_proba(X_test)
bag_df_pred_log = log_loss(y_test, bag_df_pred)
print('Bagging logloss 값:', bag_df_pred_log)
# 기본 Bagging_DecisionTree log loss 값: 1.4715524355664136

# 5) Improve Accuracy

#5.1) Grid Search

# 1단계 grid search
n_estimators = [100,300,500,700]
max_depth = [15,30,40]
rf_param_grid = dict(n_estimators=n_estimators,max_depth=max_depth)

rf= RandomForestClassifier(random_state = 777)

rf_cv = GridSearchCV(estimator=rf,
                     param_grid=rf_param_grid,
                     cv=5,
                     verbose=2,
                     n_jobs=-1)

rf_grid_result = rf_cv.fit(X_train, y_train)

#summarize results
print("Best: %f using %s" % (rf_grid_result.best_score_, rf_grid_result.best_params_))
means = rf_grid_result.cv_results_['mean_test_score']
stds = rf_grid_result.cv_results_['std_test_score']
params = rf_grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# 2단계 grid search
min_samples_leaf = [2,4,6,8]
min_samples_split = [5,10,20]
criterion = ['gini', 'entropy']
param_grid = dict(min_samples_leaf=min_samples_leaf,
                  min_samples_split= min_samples_split,
                  criterion=criterion)

rf2 = rf_grid_result.best_estimator_

rf_cv2 = GridSearchCV(estimator=rf2,
                   param_grid=param_grid,
                   cv=3,
                   verbose=2,
                   n_jobs=-1)
rf_grid_result2 = rf_cv2.fit(X_train, y_train)


# summarize results
print("Best: %f using %s" % (rf_grid_result2.best_score_, rf_grid_result2.best_params_))
means = rf_grid_result2.cv_results_['mean_test_score']
stds = rf_grid_result2.cv_results_['std_test_score']
params = rf_grid_result2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# 6) Performance of the best algorithms

# 6.1) check the performance
random_df2 = RandomForestClassifier(n_estimators = 700, max_depth = 30,
                                   random_state = 777)
random_df2.fit(X_train,y_train)
random_df_pred2 = random_df2.predict_proba(X_test)
random_df_pred_log2 = log_loss(y_test, random_df_pred2)
print('grid search 한 randomforest logloss 값:', random_df_pred_log2)
#randomforest logloss 값: 0.4007737530645949

# 6.2) futher process

# 변수 중요도 구하기
importances = random_df.feature_importances_
x_features = X_train.columns.tolist()
# 변수, 중요도를 list형태로 반환
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(x_features, importances)]
# list를 변수중요도에 따라 sorting
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('변수: {:20} 중요도의 정도: {}'.format(*pair)) for pair in feature_importances]

#psfMag_z , modelMag_i가 가장 중요한 변수

# 변수 중요도 plt 그리기
x_values = range(len(importances))
plt.figure(figsize=(20,10))
plt.bar(x_values, importances, color = 'r', edgecolor = 'k', linewidth = 1.2)
plt.xticks(x_values, x_features, rotation='vertical')
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.title('Variable Importances')

# feautre list & importance list구하기
sorted_featrues = [importance[0] for importance in feature_importances]
sorted_importances = [importance[1] for importance in feature_importances]

# importance list의 누적합
cumulative_importance = np.cumsum(sorted_importances)
# plt 그리기
plt.figure(figsize=(20,10))
plt.plot(x_values, cumulative_importance, 'b')
# 경계선 설정하기 (90%)
plt.hlines(y=0.95, xmin=0, xmax=len(sorted_importances), color='r', linestyles='dashed')
plt.xticks(x_values, sorted_featrues,rotation='vertical')
#axis label and title
plt.xlabel('Variable')
plt.ylabel('Cumulative importance')
plt.title('Cumulative importances')

#구체적으로 몇번째에서 threshold를 넘는지 확인
print('중요도 총합이 95%가 되는 feature의 갯수 :',
      np.where(cumulative_importance > 0.95 )[0][0] +1)


# 중요도 총합이 95%가 되는 상위 변수들만 추출
important_feature_names = [feature[0] for feature in feature_importances[0:16]]

important_train_features = X_train[important_feature_names]
important_test_features = X_test[important_feature_names]
print('Important train features shape:', important_train_features.shape)
print('Important test features shape:', important_test_features.shape)


# training and evaluating on important features
random_df2.fit(important_train_features,y_train)
random_df_pred3 = random_df2.predict_proba(important_test_features)
random_df_pred_log3 = log_loss(y_test, random_df_pred3)
print('importanct feautre만 포함한 randomforest logloss 값:', random_df_pred_log3)
#importanct feautre만 포함한 randomforest logloss 값: 0.4291484080235962
#더 안 좋아짐 - random_df2 이용

# 7) Fianlize Model

# 7.1) create fianal model
# plant_test 데이터 정제
plant_test = plant_test.drop(['psfMag_u','fiberID'], axis=1)

# 최종 모델 훈련
random_df2.fit(X,y)

# 7.2) predictions on test datase
# 최종 모델로 예측 데이터 생성
y_pred = random_df2.predict_proba(plant_test)

sample_submission = pd.read_csv('sample_submission.csv').set_index('id')
# submission 파일 생성
submission = pd.DataFrame(data = y_pred, columns = sample_submission.columns, index=sample_submission.index)
submission.to_csv('submission.csv', index=True)