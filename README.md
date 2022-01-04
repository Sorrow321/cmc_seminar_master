# 2022
**04.01** TODO: потыкать https://github.com/KevinMusgrave/pytorch-metric-learning

# 2021
Вникал в Deep Metric Learning, ориентируясь на <a href="https://hav4ik.github.io/articles/deep-metric-learning-survey">общую обзорную статью</a>. 
Прочитал саму статью + начал читать в глубину оригинальные статьи. На данный момент прочитал следующие:
<ul>
<li> <a href="https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31">A Discriminative Feature Learning Approach for Deep Face Recognition</a>
<li> <a href="https://arxiv.org/abs/1704.08063">SphereFace: Deep Hypersphere Embedding for Face Recognition</a>
<li> <a href="https://arxiv.org/abs/1801.07698">ArcFace: Additive Angular Margin Loss for Deep Face Recognition</a>
</ul>

Как я считаю, помимо статей часто может быть очень полезно посмотреть на топовые решения тематических соревнований.
В связи с этим составил список недавно прошедших соревнований по Deep Metric Learning. TODO: из каждого соревнования прочитать и понять топ-3 решения, выделить ключевые идеи, позволившие им выбиться в топ.
Список соревнований:
<ul>
<li> <a href="https://www.kaggle.com/c/shopee-product-matching/"> Shopee - Price Match Guarantee </a>
<li> <a href="https://www.kaggle.com/c/landmark-recognition-2021"> Google Landmark Recognition 2021 </a>
<li> <a href="https://www.drivendata.org/competitions/79/competition-image-similarity-1-dev/"> PHASE 1 | Facebook AI Image Similarity Challenge: Matching Track </a>. Топовое решение тут: https://arxiv.org/abs/2111.07090, https://github.com/WangWenhao0716/ISC-Track1-Submission
</ul>

Поскольку ArcFace на данный момент является SOTA для данной задачи, я решил его реализовать, а также попробовать применить к своей рабочей задаче (SER).<br>
Ключевая идея - ArcFace, в отличие от обычной схемы FC+SoftMax+CE, заставляет модель располагать классы достаточно далеко друг от друга, а элементы внутри классов, наоборот, более сгруппированно.<br>
Изначально это придумывалось для задач, в которых тысячи классов (такие как распознавание лиц), однако, по моей гипотезе, это может быть полезно в том случае, когда классы достаточно трудно разделимы.<br>
Реализация может быть найдена в <b>arcface.py</b> (сделаю рефакторинг репозитория чуть позже). В <b>arcface_sanity_check.py</b> - проверка на адекватность (шейпы), а также пример использования.<br><br>

Результаты на моей задаче: замена Cross Entropy на ArcFace не принесла никаких существенных изменений относительно точности на тестовой выборке. Однако эффект переобучения немного снижается (лосс на тестовой выборке меняется более плавно и более явно обнаруживается момент, когда модель начинает переобучаться). Поскольку весь весь код, относящийся к задаче, является проприетарным, возможно я переделаю бенчмарк на открытых данных (для начала MNIST и CIFAR-10, а если пойдут хорошие результаты, то можно и на ImageNet).<br><br>

UPD: Первые попытки завести на MNIST показали, что ArcFace действительно немного улучшает точность и способствует более быстрой сходимости. Однако требуется подбор параметров <b>m</b> и <b>s</b> для ArcFace. Робкое сравнение - в <b>mnist_arcface.ipynb</b>.
