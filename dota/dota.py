import pandas
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score

# Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки,
# связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).

data = pandas.read_csv('features.csv')
result_columns_names = ['duration', 'radiant_win', 'tower_status_radiant',
                        'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']
result = data[result_columns_names]
data = data.drop(result_columns_names, axis=1)

# Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число
# заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и попробуйте для
# любых двух из них дать обоснование, почему их значения могут быть пропущены.
# некоторые события не случились за 5 минут
# print(data.count()[data.count() != 97230])
# print(data.isnull().sum()[data.isnull().sum() != 0] )

# Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для
# логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание. Для
#  деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение — в этом
#  случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева. Также
#  есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании,
# но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой.
data = data.fillna(0)  # TODO check another filling of NaN

# Какой столбец содержит целевую переменную? Запишите его название. radiant_win

# Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на
# имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold),
# не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени,
# и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества. Оцените качество
# градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное
# количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). Долго ли
# настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество,
#  скорее всего, продолжит расти при дальнейшем его увеличении?

trees_count = [10, 20, 30, 100]
for index, trees in enumerate(trees_count):
    start_time = datetime.datetime.now()
    clf = GradientBoostingClassifier(verbose=0, n_estimators=trees, max_depth=2, learning_rate=0.7)  # tryed 0.2, 0.5, 0.7, 1, 2
    clf.fit(data, result['radiant_win'])
    kf = KFold(n_splits=5, shuffle=True)
    cvs = cross_val_score(estimator=clf, cv=kf, X=data, y=result['radiant_win'], scoring='roc_auc')
    print('Trees = ', trees, 'Cross value score = ', cvs.mean())
    print('Time elapsed:', datetime.datetime.now() - start_time)

# Trees =  10 Cross value score =  0.6645850512249734
# Time elapsed: 0:00:32.825878
# Trees =  20 Cross value score =  0.6812031672749024
# Time elapsed: 0:01:01.634525
# Trees =  30 Cross value score =  0.6901283769279053
# Time elapsed: 0:01:28.898085
# Trees =  100 Cross value score =  0.7062319153700038
# Time elapsed: 0:04:45.149309

# print(data['match_id'].value_counts())
