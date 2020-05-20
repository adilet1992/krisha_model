import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.externals import joblib

df = pd.read_csv('data.csv')

#Normalization of data
def norm_price(price):
    a = ''
    for s in price.split():
        a = a + s
    return int(a)
df = df[df.floor.str.contains('из')]
df['count'] = df.city.apply(lambda x:len(x.split(',')))
df = df[df['count'] == 2]
df['count2'] = df.house.apply(lambda x:len(x.split(',')))
df = df[df['count2'] == 2]
df.price = df.price.apply(lambda x:x.strip())
df.title = df.title.apply(lambda x:x.strip())
df['start_floor'] = df.floor.apply(lambda x:int(x.split(' ')[0]))
df['end_floor'] = df.floor.apply(lambda x:int(x.split(' ')[2]))
df.price = df.price.apply(lambda x:norm_price(x))
df['district'] = df.city.apply(lambda x:x.split(',')[1].split(' ')[1].strip())
df['rooms'] = df.title.apply(lambda x:int(x.split('-')[0]))
df.area = df.area.apply(lambda x:float(x.split(' ')[0]))
df['material'] = df.house.apply(lambda x:x.split(',')[0])
df['year'] = df.house.apply(lambda x:int(x.split(',')[1].split('г.п.')[0].strip()))
df = df[['area', 'start_floor', 'end_floor', 'district', 'rooms', 'material', 'year', 'price']]

df2 = df
df2 = df2.drop_duplicates(['area', 'start_floor', 'end_floor', 'district', 'rooms', 'material', 'year'], keep='last')
q_area = np.quantile(df2.area, [0.1, 0.9])
df2['sq_price'] = df2.price/df2.area
q_price = np.quantile(df2.price, [0.1, 0.9])
q_sq_price = np.quantile(df2['sq_price'], [0.1, 0.9])
df2 = df2[(df2.area >= q_area[0])&(df2.area <= q_area[1])]
df2 = df2[(df2.price >= q_price[0])&(df2.price <= q_price[1]*1.5)]
df2 = df2[(df2.sq_price >= q_sq_price[0])&(df2.sq_price <= q_sq_price[1]*1.5)]
df2 = df2[(df2.material != 'иное')&(df2.year >= 1960)&(df2.end_floor <= 35)&(df2.rooms <= 10)]
df2 = df2[df2.start_floor <= df2.end_floor]
df2 = df2[df2.end_floor >= 2]
df2 = df2[df2.year <= 2020]
df2_panel = df2[df2.material == 'панельный']
df2_kirpich = df2[df2.material == 'кирпичный']
df2_monolit = df2[df2.material == 'монолитный']
df2_panel = df2_panel[df2_panel.year <= 1996]
df2_kirpich = df2_kirpich[df2_kirpich.year <= 2004]
df2_monolit = df2_monolit[df2_monolit.year >= 2004]
df3 = pd.concat([df2_panel, df2_kirpich, df2_monolit])

#Feature engineering
df2['floor_coef'] = (df2.start_floor - 1)/(df2.end_floor - 1)
df2['avg_sq_room'] = df2.area/df2.rooms
df2['oldness'] = 2020 - df2['year']

#Training
df3 = pd.get_dummies(df2)
df3 = df3.sample(frac=1).reset_index(drop=True)
X = df3.drop(['sq_price', 'price'], axis=1)
y = df3['price']

forest = RandomForestRegressor()
boost = GradientBoostingRegressor()
extra = ExtraTreesRegressor()
grid_forest = GridSearchCV(estimator=forest, cv=5, param_grid={'n_estimators':[80, 100, 120], 'max_depth':[3, 5, 7]})
grid_forest.fit(X, y)

grid_boost = GridSearchCV(estimator=boost, cv=5, param_grid={'n_estimators':[80, 100, 120], 'max_depth':[3, 5, 7], 
                                                             'learning_rate':[0.1, 0.01, 0.001]})
grid_boost.fit(X, y)
grid_extra = GridSearchCV(estimator=extra, cv=5, param_grid={'n_estimators':[80, 100, 120], 'max_depth':[3, 5, 7]})
grid_extra.fit(X, y)

forest = RandomForestRegressor(n_estimators=grid_forest.best_params_['n_estimators'], max_depth=grid_forest.best_params_['max_depth'])
boost = GradientBoostingRegressor(n_estimators=grid_boost.best_params_['n_estimators'], max_depth=grid_boost.best_params_['max_depth'],
                                 learning_rate = grid_boost.best_params_['learning_rate'])
extra = ExtraTreesRegressor(n_estimators=grid_extra.best_params_['n_estimators'], max_depth=grid_extra.best_params_['max_depth'])
voter = VotingRegressor(estimators=[('forest', forest), ('boost', boost), ('extra', extra)])
model = voter.fit(X, y)

joblib.dump(model, 'model.pkl')