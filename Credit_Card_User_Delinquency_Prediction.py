#----------------***************************!중요중요중요!****************----------------------------------------------
# Python 3.9.7 version
# tensorflow 2.5.0 version
# keras 2.5.0 version
import os
os.chdir(r'C:\동근\DACON\신용카드 사용자 연체 예측')

# 데이터 불러오기
import pandas as pd
df = pd.read_csv('train.csv')

# 데이터 확인
df.info()
df.shape
df.describe()

###-----------데이터 정제----------------------------------------------------------------------
# 인덱스는 필요없으므로 제거
df.drop('index',axis=1, inplace=True)

# 다중공선성 확인
# child_num과 family_size의 상관성이 높음 -> child_num 제거
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize =(9,9))
corr = df.corr()
sns.heatmap(corr,cmap='RdBu')

df.drop('child_num',axis=1, inplace=True)

# 'FLAG_MOBIL'는 단일 값을 가지므로 제거
df['FLAG_MOBIL'].value_counts()
df.drop('FLAG_MOBIL',axis=1, inplace=True)

# 범주형인데 숫자형으로 되어 있는 피처 범주형으로 변환
df = df.astype({'work_phone':'object', 'phone':'object', 'email':'object'})

# 문자형인 이진데이터 숫자(0, 1)로 변경
df['gender'] = df['gender'].replace(['F','M'],[0,1])
df['car'] = df['car'].replace(['N','Y'],[0,1])
df['reality'] = df['reality'].replace(['N','Y'],[0,1])

# 이진 데이터는 숫자형으로 변환
df = df.astype({'work_phone':'float', 'phone':'float', 'email':'float','gender':'float','car':'float','reality':'float'})
# 데이터가 직업 유형이므로 null값을 'No job'으로 변환
df.isnull().sum()
df = df.fillna({'occyp_type':'No job'})
df['occyp_type'].value_counts()

# DAYS_BIRTH 변환작업
# DAYS_BIRTH가 태어난 후 부터 -(마이너스)로 일수 계산으로 되어있기 때문에
# abs()함수로 +로 만들어준 후 365.25(윤년)로 나누어 나이 계산
df['DAYS_BIRTH'].head()
import numpy as np
df['Age'] = round(abs(df['DAYS_BIRTH'])/365.25,0).astype(np.float32)
df['Age'].head()
df.drop('DAYS_BIRTH', axis=1, inplace=True)

# 업무 시작일(DAYS_EMPLOYED) 변환(Year)
df['DAYS_EMPLOYED'].head()
df['YEARS_EMPLOYED'] = [0 if s >=0 else round(abs(s)/365.25,2) for s in df['DAYS_EMPLOYED']]
df['YEARS_EMPLOYED'].head()
df.drop('DAYS_EMPLOYED', axis=1, inplace=True)

# 신용카드 발급 월(begin_month) 변환, 년으로 만들어줌
df['begin_month'].head()
df['begin_year'] = [0 if s >=0 else round(abs(s)/12,2) for s in df['begin_month']]
df['begin_year'].head()
df.drop('begin_month', axis=1, inplace=True)

# 수치형 데이터 이상치 제거를 위한 확인
num_col = ['income_total',
           'family_size',
           'Age',
           'YEARS_EMPLOYED',
           'begin_year']
for i in num_col:
    print(i)
    print(df[i].value_counts())
    print('\n')
    

## family_size(가족 수)가 20인 값 1개, 15인 값 3개가 있음
# 위의 상황 이상치라 판단하고 제거
df.drop(index = df[df['family_size']==20].index,axis='index' ,inplace=True)
df.drop(index = df[df['family_size']==15].index,axis='index', inplace=True)
df['family_size'].value_counts()  
df.info()
df.tail()


# 수치형 데이터 정규화
from sklearn.preprocessing import MinMaxScaler
num_col = ['income_total',
           'family_size',
           'Age',
           'YEARS_EMPLOYED',
           'begin_year']
# MinMaxScaler객체 생성
scaler = MinMaxScaler()
# MinMaxScaler 로 데이터 셋 변환. fit() 과 transform() 호출.  
scaler.fit(df.loc[:,num_col])
df_scaled = scaler.transform(df.loc[:,num_col])

# transform()시 scale 변환된 데이터 셋이 numpy ndarry로 반환되어 이를 DataFrame으로 변환
df_scaled = pd.DataFrame(data=df_scaled, columns=num_col) 
df_scaled.head()
df_scaled.tail()
df_scaled.info()
print('feature들의 최소 값')
print(df_scaled.min())
print('\nfeature들의 최대 값')
print(df_scaled.max())
df.drop(num_col, axis=1, inplace=True)
df_scaled.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)

df = pd.concat((df, df_scaled), axis=1)
df.isnull().sum()
df.info()


"""
# 수치형 데이터 표준화
from sklearn.preprocessing import StandardScaler
num_col = ['income_total',
           'family_size',
           'Age',
           'YEARS_EMPLOYED',
           'begin_year']
scaler = StandardScaler()

scaler.fit(df.loc[:,num_col])
df_scaled = scaler.transform(df.loc[:,num_col])
df_scaled = pd.DataFrame(data=df_scaled, columns=num_col)
df.drop(num_col, axis=1, inplace=True)
df = pd.concat((df, df_scaled), axis=1) 
df.describe()
"""
# 범주형 데이터 원핫 인코딩
from sklearn.preprocessing import OneHotEncoder
# 이진데이터 'gender','car','reality','work_phone','phone','email'
onehot_col = [
              'income_type',
              'edu_type',
              'family_type',
              'house_type',
              'occyp_type'
             ]

onehot_encoder = OneHotEncoder()
onehot_encoder.fit(df.loc[:,onehot_col])
train_onehot_df = pd.DataFrame(onehot_encoder.transform(df.loc[:,onehot_col]).toarray(), 
                               columns=onehot_encoder.get_feature_names(onehot_col))
train_onehot_df.info()
df.drop(onehot_col, axis=1, inplace=True)
df_scaled.reset_index(drop=True,inplace=True)
df.reset_index(drop=True,inplace=True)

df = pd.concat([df, train_onehot_df], axis=1)
df.isnull().sum()
df.info()

# 피처와 레이블 나누기
y_target = df['credit']
X_features = df.drop('credit', axis=1, inplace=False)

# 데이터셋 80%, 테스트셋 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_features, y_target, test_size=0.2,stratify=y_target)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,stratify=y_train)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
X_val.shape
y_val.shape

# 딥러닝 modeling
from keras.models import Sequential
model = Sequential()

from keras.layers import Dense
# 첫번째 은닉 레이어
model.add(Dense(114, activation='relu',input_dim=51))

# 두번째 은닉 레이어
model.add(Dense(100, activation='relu'))

# 세번째 은닉 레이어
model.add(Dense(10, activation='relu'))

# 출력 레이어, 출력층이 3개이기 때문에 다층 분류이므로 softmax 사용
model.add(Dense(3,activation='softmax'))

# model compile
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 반복 훈련 
history = model.fit(X_train,y_train,validation_data=(X_val, y_val),epochs=100,batch_size=0)

# 테스트 정확도
scores = model.evaluate(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores[1]*100))

scores = model.evaluate(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores[1]*100))

# 훈련 과정 시각화 (정확도)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# 훈련 과정 시각화 (손실)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

"""
숫자형 데이터만 포함했을 때 train 정확도: 64.03%
숫자형 데이터만 포함했을 때 test 정확도: 64.55%

모든 데이터(범주형은 인코딩) 포함했을 때 train 정확도: 64.46%
모든 데이터(범주형은 인코딩) 포함했을 때 test 정확도: 62.83%

income_total 제외 (범주형은 인코딩) 했을 때 train 정확도: 72.80%
income_total 제외 (범주형은 인코딩) 했을 때 test 정확도: 68.41%

수치형 스케일링(정규화), 범주형 원핫인코딩 했을 때 train 정확도: 86.27%
수치형 스케일링(정규화), 범주형 원핫인코딩 했을 때 test 정확도: 65.51%

1은닉 100, 2은닉 100, train 정확도: 74.62%
1은닉 100, 2은닉 100, test 정확도: 67.46%
"""
# 로지스틱
from sklearn.linear_model import LogisticRegression 

model = LogisticRegression()

model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.metrics import classification_report # 정밀도, 재현율 확인
y_pred=model.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.model_selection import GridSearchCV
params={'penalty':['l2', 'l1'],
        'C':[0.01, 0.1, 1, 1, 5, 10, 100],
        'max_iter' : [100,1000,2000]}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

# SVM
from sklearn.svm import SVC
model = SVC() # SVM 모델 생성
model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.model_selection import GridSearchCV
params={'C': [0.001, 0.01, 0.1, 1, 10, 100],
             'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

# tree
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.model_selection import GridSearchCV
params={'max_depth': [1, 2, 3, 4, 5],
        'max_leaf_nodes': list(range(2, 100)),
        'min_samples_split': [2, 3, 4]}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))


# knn
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.model_selection import GridSearchCV
params={'n_neighbors' : list(range(1,20)),
    'weights' : ['uniform', 'distance'],
    'metric' : ['euclidean', 'manhattan', 'minkowski']}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

# randomforest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.model_selection import GridSearchCV  # 여기부터 다시 시작
params={
    'n_estimators':[100,200],
    'max_depth':[6,8,10,12],
    'min_samples_leaf':[8,12,18],
    'min_samples_split':[3,5,8,16,20]
}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))

# GBM
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
scores = model.score(X_train, y_train)
print('Training Accuracy: %.2f%%\n' %(scores*100))
scores = model.score(X_test, y_test)
print('Testing Accuracy: %.2f%%\n' %(scores*100))

from sklearn.model_selection import GridSearchCV
params={
    'n_estimators' : [100,200,300,400,500],
    'learning_rate' : [0.01,0.05,0.1,0.15]
}
grid_clf = GridSearchCV(model, param_grid=params, scoring='accuracy', cv=5 )
grid_clf.fit(X_val, y_val)
print('최적 하이퍼 파라미터:{0}, 최적 평균 정확도:{1:.3f}'.format(grid_clf.best_params_, grid_clf.best_score_))


