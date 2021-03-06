# 2022
**07.05** Поскольку чаще всего в статьях тестируют на CUB-200 версии 2011 года, пересчитал модели на этом датасете, заменил в тексте. Также более точно оценил пропускную способность моделей.  
**06.05** Залил первую версию текста. Ссылка на логи экспериментов: https://wandb.ai/ilya_fedorov/my-test-project/overview?workspace=user-ilya_fedorov. Не успел сделать: тестирование на Google Landmark Retrieval (проблема в том, что evaluation доступ только через кагл, орги не выложили тестовый датасет в публичный доступ), а также ArcFace (в реализации был баг с моей стороны, в итоге все эксперименты оказались невалидным... если останется время, может быть, дополнительно прогоню эксперименты на arcface).  
**02.05** Провожу эксперименты, подбираю параметры. Побил sota на CUB200 с помощью Swin Tranformer (может, будет успех и на других датасетах). Начал писать текст.  
**20.04** Написал общий пайплайн для обучения и тестирования. Логирование идет в wandb (возможно, сделаю его публичным и опубликую сюда ссылку). Результат обучения на датасете SOP приблизительно совпадает с результатами, описанными в статье https://arxiv.org/abs/2102.05644, так что все работает корректно. Осталось провести эксперименты для всех лоссов, датасетов и моделей. Также пришла идея добавить модель MLP Mixer для тестирования (что позволит сравнить 3 полностью разных архитектуры: CNN-based, Transformer-based и MLP-based). Помимо этого можно попробовать сделать визуализацию частей картинки, на которые "смотрит" модель в разных случаях. Текст курсовой начну писать приблизительно в начале мая, сейчас основная задача - получить результаты.  
**04.04** Изучил способы evalution'а для image retrieval. Глобально есть 2 варианта: когда в тестовом датасете index отделен от query или совпадает с ним. Датасеты с отделенным индексом: In-Shop, GLDv2. С совпадающим: Cub-200, SOP.  
**26.03**
Имплементировал (собрал из разных библиотек) все указанные ниже 4 модели. К Swin V2 еще не выложили в публичный доступ веса, но есть код, так что если останется время, то можно будет попробовать обучить его с нуля, согласно оригинальной авторской статье.  
Согласовал со своей компанией, что смогу немножко поюзать мощности (2-4-8 видеокарт, но без фанатизма).  
Выбрал 4 датасета на данный момент. Из них GLDV2 - большой датасет (на 100 гб+), остальные меньше.  
Для создания пайплайнов metric learning будет использовать эту библиотеку: https://github.com/KevinMusgrave/pytorch-metric-learning.  
На данный момент цель работы - сравнить разные архитектуры (в основном CNN-Based vs Transformer-Based) на разных датасетах, используя разные лоссы (проверить Contrastive Loss, Triplet Loss, ArcFace - как минимум). В идеале, если получится обучить веса для Swin v2 и сравнить и его, то это будет (кажется) первая работа, тестирующая для deep metric learning эту архитектуру.  
TODO на ближайшую неделю - имплементировать пайплайны для обучения моделей, а также удобного трекинга экспериментов. Скорей всего для трекинга буду использовать Weights & Biases.  
**23.03**
Модели на тест: 
- EfficientNet
- EfficientNet V2
- ViT
- Swin
- (со звездочкой) Swin V2
- Список будет пополняться

Датасеты на тест:
- Google Landmarks Dataset v2
- Cub 200
- Stanford Online Product
- In-Shop
- Список будет пополняться
  
**21.03** Как я понял, веса Swin v2 еще не скинули в публичный доступ (хотя статья с осени). Пока читаю похожие статьи (по типу https://arxiv.org/abs/2102.05644), чтобы выбрать список датасетов. Помимо популярных, возьму также датасет из прошлогоднего соревнования по image retrieval: https://www.kaggle.com/c/landmark-retrieval-2021/overview/description, тк топовые решения публичные и по ним есть выступления. Но там ансамбль из кучи моделей, так что скорей всего побить их скор не получится (да и бессмысленно). Но почерпнуть какие-то мысли я думаю будет полезно. Пока что вырисовываю такое исследование: сравнение Resnet vs Effnet vs ViT vs Swin с разными лоссами на разных датасетах + использую некоторые лайфхаки из других статей по смежной теме.    
**19.03** Определился с темой: сравнение современных нейросетевых архитектур для deep metric learning computer vision (точное название подредактирую позже). Идея в том, чтобы взять последние CNN (ResNet, EfficientNet) и сети на основе трансформеров (ViT, SWin, SWin 2) и провести их сравнение на популярных датасетах для metric learning. Для повышения точности подсмотреть разные идеи из топовых решений последних соревнований по теме метрик лернинга. На данный момент заканчиваю изучать архитектуры (чтобы понимать о чем речь вообще), и скоро начну писать код. Про мощности узнаю на работе - если не выдадут, то арендую.  
**10.03** Из-за начала войны немножко выбился из колеи. Начинаю в ближайшее время активно писать курсовую.  
**14.02** Изучаю, что можно сделать в качестве исследования. Поспрашивал у людей, выдали такой список статей на почитать:
Transformers for metric learning  
https://arxiv.org/abs/2102.05644  
https://proceedings.neurips.cc/paper/2021/hash/0f49c89d1e7298bb9930789c8ed59d48-Abstract.html  
https://www.bmvc2021-virtualconference.com/assets/papers/1551.pdf  
https://arxiv.org/abs/2109.12564  
https://openaccess.thecvf.com/content/WACV2022/html/Song_All_the_Attention_You_Need_Global-Local_Spatial-Channel_Attention_for_Image_WACV_2022_paper.html  
https://openaccess.thecvf.com/content/WACV2022/html/Black_Visualizing_Paired_Image_Similarity_in_Transformer_Networks_WACV_2022_paper.html  

Разного рода hard example mining, hard pair mining (это даже важнее лоссов обычно)  
http://openaccess.thecvf.com/content_CVPR_2019/html/Suh_Stochastic_Class-Based_Hard_Example_Mining_for_Deep_Metric_Learning_CVPR_2019_paper.html  
http://openaccess.thecvf.com/content_cvpr_2018_workshops/w1/html/Smirnov_Hard_Example_Mining_CVPR_2018_paper.html  
http://openaccess.thecvf.com/content_iccv_2017/html/Harwood_Smart_Mining_for_ICCV_2017_paper.html  
http://openaccess.thecvf.com/content_ICCV_2017_workshops/w27/html/Smirnov_Doppelganger_Mining_for_ICCV_2017_paper.html  

Knowledge Distillation for metric learning (уже есть всякие методы, но можно придумать ещё) 

Mixup/CutMix-like methods, suitable for metric learning (в стандартном варианте непонятно как "смешивать таргеты", если картинки из разных классов)  
https://proceedings.neurips.cc/paper/2020/hash/f7cade80b7cc92b991cf4d2806d6bd78-Abstract.html  
https://arxiv.org/abs/2106.04990  
https://arxiv.org/abs/2010.08887  

Методы metric learning, стойкие к noisy labels (а то это становится проблемой, особенно когда ещё hard example mining используется, и получается, что выбираемые "сложные примеры" - это примеры с неправильными лейблами)

Генерация "виртуальных классов" и "виртуальных примеров"
https://proceedings.neurips.cc/paper/2018/hash/d79aac075930c83c2f1e369a511148fe-Abstract.html  
http://openaccess.thecvf.com/content/ICCV2021/html/Ko_Learning_With_Memory-Based_Virtual_Classes_for_Deep_Metric_Learning_ICCV_2021_paper.html  
https://www.aaai.org/AAAI21Papers/AAAI-1275.GuG.pdf  
http://openaccess.thecvf.com/content/CVPR2021/html/Li_VirFace_Enhancing_Face_Recognition_via_Unlabeled_Shallow_Data_CVPR_2021_paper.html  
https://arxiv.org/abs/2201.01008  
https://www.sciencedirect.com/science/article/pii/S0031320320304465?  casa_token=ckxSbwsZVf0AAAAA:y64cyBSBe1e7dQG_8vm0bTGzYkE6b2B24VXPVriZh6_zggUIBs_zNiV6Q2S_ZS9R9uud4FA  

Разного рода Cross-Batch Memory, когда в процессе обучения запоминаются эмбеддинги (последние или оказавшиеся наиболее сложными), которые потом используются для уточнения "тренировочного сигнала".  
http://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html  
https://arxiv.org/abs/2008.09809  
https://arxiv.org/abs/2008.06674  
https://arxiv.org/abs/2105.02103  

Fairness and bias in metric learning
https://dl.acm.org/doi/abs/10.1145/3457607  

Помимо этого также смотрю исследования на тему self-supervised learning

**01.02** Опубликовал статью на Medium, описывающую данный пет проект: https://medium.com/@bultiman/creating-a-deep-learning-application-telling-which-celebrity-your-voice-sounds-like-e28a49117f61<br>
**28.01** Идея небольшого пет проекта. Взять модель для верификации спикера (к примеру, эту https://huggingface.co/microsoft/unispeech-sat-base-plus-sv). Она обучена на метрик лернинг лосс Additive Margin Softmax loss на датасете VoxCeleba1 (голоса разных знаменитостей). Идея: прогоняем весь VoxCeleba1 датасет через эту сетку, загоняем в qdrant весь датасет эмбеддингов. А далее юзер можно поиграть в игру "на кого из знаменитостей больше всего похож мой голос". <a href="https://github.com/Sorrow321/celeba_similarity">Репозиторий пет проекта.</a><br>
**12.01** TODO: потыкать https://github.com/qdrant/qdrant. Аналог FAISS (и http://milvus.io/) на Rust. Надо сделать пример проекта на базе этого движка.<br>
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
