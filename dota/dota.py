import pandas
import time
import datetime
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
# Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки,
# связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).

data = pandas.read_csv('features.csv')
result_columns_names = ['duration', 'radiant_win', 'tower_status_radiant',
                        'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
result = data[result_columns_names]
data = data.drop(result_columns_names, axis=1)

# # Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число
# # заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и попробуйте для
# # любых двух из них дать обоснование, почему их значения могут быть пропущены.
# # некоторые события не случились за 5 минут
# # print(data.count()[data.count() != 97230])
# # print(data.isnull().sum()[data.isnull().sum() != 0] )
#
# # Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для
# # логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание. Для
# #  деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение — в этом
# #  случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева. Также
# #  есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании,
# # но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.
data = data.fillna(0)  # TODO check another filling of NaN
#
# # Какой столбец содержит целевую переменную? Запишите его название. radiant_win
#
# # Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на
# # имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# # не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# # и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества. Оцените качество
# # градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное
# # количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). Долго ли
# # настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество,
# #  скорее всего, продолжит расти при дальнейшем его увеличении?
#
# trees_count = [10, 20, 30, 100]
# for index, trees in enumerate(trees_count):
#     start_time = datetime.datetime.now()
#     clf = GradientBoostingClassifier(verbose=0, n_estimators=trees, max_depth=2, learning_rate=0.7)  # tryed 0.2, 0.5, 0.7, 1, 2
#     clf.fit(data, result['radiant_win'])  # fir doesn't need here cvs creates copy
#     kf = KFold(n_splits=5, shuffle=True)
#     cvs = cross_val_score(estimator=clf, cv=kf, X=data, y=result['radiant_win'], scoring='roc_auc')
#     print('Trees = ', trees, 'Cross value score = ', cvs.mean())
#     print('Time elapsed:', datetime.datetime.now() - start_time)
#
# # Trees =  10 Cross value score =  0.6645850512249734
# # Time elapsed: 0:00:32.825878
# # Trees =  20 Cross value score =  0.6812031672749024
# # Time elapsed: 0:01:01.634525
# # Trees =  30 Cross value score =  0.6901283769279053
# # Time elapsed: 0:01:28.898085
# # Trees =  100 Cross value score =  0.7062319153700038
# # Time elapsed: 0:04:45.149309


# # Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей
# # идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero,
# # d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой
# # выборке с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?
# # не изменилось
# kategorial_columns = ['lobby_type',
#                       'r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
#                       'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
# data = data.drop(kategorial_columns, axis=1)


# # Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с помощью
# # кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. Подберите при этом лучший
# # параметр регуляризации (C). Какое наилучшее качество у вас получилось? Как оно соотносится с качеством градиентного
# #  бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с
# # градиентным бустингом?
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# # regularization = np.arange(1, 10, 0.1)  # 8.1 max Max is  0.7164357243534086 With C =  1.3000000000000003
# regularization = np.power(10.0, np.arange(-5, 6))  # Max is  0.7164706557971849 With C =  0.01
# max = 0
# max_c = 0
# for i, C in enumerate(regularization):
#     start_time = datetime.datetime.now()
#     # clf = LogisticRegression(verbose=0, penalty='l2', C=C, n_jobs=-1)
#     clf = LogisticRegression(verbose=0, C=C)
#     kf = KFold(n_splits=5, shuffle=True, random_state=45)
#     cvs = cross_val_score(estimator=clf, cv=kf, X=data, y=result['radiant_win'], scoring='roc_auc')
#     # print('Time elapsed:', datetime.datetime.now() - start_time)
#     print('C is ', C, 'Cross value score ', cvs.mean())
#     if cvs.mean() > max:
#         max = cvs.mean()
#         max_c = C
# print('Max is ', max, 'With C = ', max_c)

# # На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои
# # играли за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают
# # чаще, чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может
# # пригодиться фукнция unique или value_counts).  #108
# heroes_columns = ['r1_hero', 'r2_hero', 'r3_hero', 'r4_hero', 'r5_hero',
#                   'd1_hero', 'd2_hero', 'd3_hero', 'd4_hero', 'd5_hero']
# print(data[heroes_columns].nunique())

# Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных
# героев. Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице,
# если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire. Ниже вы можете найти
# код, который выполняет данной преобразование. Добавьте полученные признаки к числовым, которые вы использовали во
# втором пункте данного этапа.
# N — количество различных героев в выборке

heroes_lenth = pandas.read_csv('heroes.csv').shape[0]
X_pick = np.zeros((data.shape[0], heroes_lenth))
for i, match_id in enumerate(data.index):
    for p in range(5):
        X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

data_heroes = pandas.DataFrame(X_pick)   # not so elegant but works
data = pandas.concat([data, data_heroes], axis=1)

scaler = StandardScaler()
data = scaler.fit_transform(data)
# regularization = np.arange(1, 10, 0.1)  # 8.1 max Max is  0.7164357243534086 With C =  1.3000000000000003
regularization = np.power(10.0, np.arange(-5, 6))  # Max is  0.7164706557971849 With C =  0.01
max = 0
max_c = 0
# for i, C in enumerate(regularization):
#     start_time = datetime.datetime.now()
#     # clf = LogisticRegression(verbose=0, penalty='l2', C=C, n_jobs=-1)
#     clf = LogisticRegression(verbose=0, C=C)
#     kf = KFold(n_splits=5, shuffle=True, random_state=45)
#     cvs = cross_val_score(estimator=clf, cv=kf, X=data, y=result['radiant_win'], scoring='roc_auc')
#     # print('Time elapsed:', datetime.datetime.now() - start_time)
#     print('C is ', C, 'Cross value score ', cvs.mean())
#     if cvs.mean() > max:
#         max = cvs.mean()
#         max_c = C
# print('Max is ', max, 'With C = ', max_c)  # Max is  0.7519233522872815 With C =  0.01e

clf = LogisticRegression(verbose=0, penalty='l2', C=0.01)
model = clf.fit(data, result['radiant_win'])

test_data = pandas.read_csv('features_test.csv')
test_data = test_data.fillna(0)


X_pick = np.zeros((test_data.shape[0], heroes_lenth))
for i, match_id in enumerate(test_data.index):
    for p in range(5):
        X_pick[i, test_data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, test_data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
data_heroes = pandas.DataFrame(X_pick)   # not so elegant but works
test_data = pandas.concat([test_data, data_heroes], axis=1)
test_data = scaler.fit_transform(test_data)

predictions = model.predict_proba(test_data)
print(predictions)

print('Max: ', predictions.max())
print('Min: ', predictions.min())
