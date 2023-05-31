#----------------***************************!중요중요중요!****************----------------------------------------------
# Python 3.9.7 version
import os
os.chdir(r'C:\동근\DACON\신용카드 사용자 연체 예측')

# 데이터 불러오기
import pandas as pd
df = pd.read_csv('train.csv')

### Categorical Variable EDA
## 신용등급비율
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
# 한글깨짐 오류 해결
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False

# credit(신용도) 비율
plt.subplots(figsize = (8,8))
plt.pie(df['credit'].value_counts(), labels = df['credit'].value_counts().index, 
        autopct="%.2f%%", shadow = True, startangle = 90)
plt.title('신용 등급 비율', size=20)
plt.show()

# 등급에 따른 차이를 보기 위한 데이터 분류
train_0 = df[df['credit']==0.0]
train_1 = df[df['credit']==1.0]
train_2 = df[df['credit']==2.0]

# Categorical 그래프 함수 정의
import seaborn as sns
def cat_plot(column):

  f, ax = plt.subplots(1, 3, figsize=(16, 6))


  sns.countplot(x = column,
                data = train_0,
                ax = ax[0],
                order = train_0[column].value_counts().index)
  ax[0].tick_params(labelsize=12)
  ax[0].set_title('credit = 0')
  ax[0].set_ylabel('count')
  ax[0].tick_params(rotation=50)


  sns.countplot(x = column,
                data = train_1,
                ax = ax[1],
                order = train_1[column].value_counts().index)
  ax[1].tick_params(labelsize=12)
  ax[1].set_title('credit = 1')
  ax[1].set_ylabel('count')
  ax[1].tick_params(rotation=50)

  sns.countplot(x = column,
                data = train_2,
                ax = ax[2],
                order = train_2[column].value_counts().index)
  ax[2].tick_params(labelsize=12)
  ax[2].set_title('credit = 2')
  ax[2].set_ylabel('count')
  ax[2].tick_params(rotation=50)
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  plt.show()
  
## 신용등급에 따른 성별 차이
# 모든 등급에서 남성보다 여성이 더 많음
cat_plot("gender")

## 신용등급에 따른 차량 소유 차이
# 모든 등급에서 차를 보유하고 있지 않은 고객들이 많음
cat_plot('car') 

## 신용등급에 따른 부동산 소유 차이
# 모든 등급에서 부동산을 소유한 사람들이 많았음
cat_plot('reality') 

## 신용등급에 따른 소득 분류 차이
# 높은 신용(credit=0)에서는 학생 존재X
cat_plot('income_type') 

## 신용등급에 따른 교육 수준 차이
# 모든 등급에서 교육 수준의 순위가 같음
cat_plot('edu_type')

## 신용등급에 따른 결혼 여부 차이
# 모든 등급에서 결혼 한 사람이 많음
cat_plot('family_type')

## 신용등급에 따른 생활 방식의 차이
# 집/아파트에 사는 사람이 많음
cat_plot('house_type')

## 신용등급에 따른 핸드폰 소지 차이
# 모든 사람이 핸드폰 소지
cat_plot('FLAG_MOBIL')

## 신용등급에 따른 업무용 전화 소유 차이
# 업무용 전화를 소유하지 않는 사람이 많음, 핸드폰이 있기 때문?
cat_plot('work_phone')

## 신용등급에 따른 가정용 전화 소유 차이
# 가정용 전화를 소유하지 않는 사람이 많음, 핸드폰이 있기 때문?
cat_plot('phone')

## 신용등급에 따른 이메일 소유 차이
# 모든 등급에서 이메일을 소유한 사람이 많음
cat_plot('email')

## 신용등급에 따른 직업 유형 차이(결측치는 'No Job'으로 대체)
# 등급별로 직업 유형의 순위 차이 있음
train_0 = train_0.fillna({'occyp_type':'No job'})
train_1 = train_1.fillna({'occyp_type':'No job'})
train_2 = train_2.fillna({'occyp_type':'No job'})

f, ax = plt.subplots(1, 3, figsize=(16, 6))
sns.countplot(y = 'occyp_type', data = train_0, order = train_0['occyp_type'].value_counts().index, ax=ax[0])
sns.countplot(y = 'occyp_type', data = train_1, order = train_1['occyp_type'].value_counts().index, ax=ax[1])
sns.countplot(y = 'occyp_type', data = train_2, order = train_2['occyp_type'].value_counts().index, ax=ax[2])
plt.subplots_adjust(wspace=0.5, hspace=0.3)
plt.show()

# Numerical 그래프 함수 정의
def num_plot(column):
  
  fig, axes = plt.subplots(1, 3, figsize=(16, 6))


  sns.distplot(train_0[column],
                ax = axes[0])
  axes[0].tick_params(labelsize=12)
  axes[0].set_title('credit = 0')
  axes[0].set_ylabel('count')

  sns.distplot(train_1[column],
                ax = axes[1])
  axes[1].tick_params(labelsize=12)
  axes[1].set_title('credit = 1')
  axes[1].set_ylabel('count')

  sns.distplot(train_2[column],
                ax = axes[2])
  axes[2].tick_params(labelsize=12)
  axes[2].set_title('credit = 2')
  axes[2].set_ylabel('count')
  plt.subplots_adjust(wspace=0.3, hspace=0.3)
  
## 신용등급에 따른 자녀 수의 차이
# 별 차이 없어 보임
num_plot("child_num")

## 신용등급에 따른 신용카드 발급 년(Year) 차이
# 발급 받은지 1년정도 되는 사람이 가장 많다
train_0['new_begin_month'] = [0 if s >=0 else round(abs(s)/12,2) for s in train_0['begin_month']]
train_1['new_begin_month'] = [0 if s >=0 else round(abs(s)/12,2) for s in train_1['begin_month']]
train_2['new_begin_month'] = [0 if s >=0 else round(abs(s)/12,2) for s in train_2['begin_month']]
num_plot("new_begin_month")

## 신용등급에 따른 연간 소득 차이
# 별 차이 없어 보임
num_plot("income_total")

## 신용등급에 따른 연령대 차이
# 주로 30, 40대가 많아 보임, 20대는 적음
import numpy as np
train_0['new_age'] = round(abs(train_0['DAYS_BIRTH'])/365.25,0).astype(np.int32)
train_1['new_age'] = round(abs(train_1['DAYS_BIRTH'])/365.25,0).astype(np.int32)
train_2['new_age'] = round(abs(train_2['DAYS_BIRTH'])/365.25,0).astype(np.int32)
num_plot('new_age')

## 신용등급에 따른 업무 기간 차이
# 양수는 고용되지 않은 상태이므로 0으로 대체
# 직업이 없는 사람이 많음
train_0['worked_year'] = [0 if s >=0 else round(abs(s)/365.25,2) for s in train_0['DAYS_EMPLOYED']]
train_1['worked_year'] = [0 if s >=0 else round(abs(s)/365.25,2) for s in train_1['DAYS_EMPLOYED']]
train_2['worked_year'] = [0 if s >=0 else round(abs(s)/365.25,2) for s in train_2['DAYS_EMPLOYED']]
num_plot('worked_year')

## 신용등급에 따른 가족 수 차이
# 가족 수가 2명이 사람이 많음, credit=2에서는 비교적 왼쪽으로 치우침
num_plot('family_size')
