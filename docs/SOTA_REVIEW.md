Ниже — итоговый обзор по состоянию на **22 сентября 2026 г.** Я трактовал постановку именно в заданном вами смысле: `?` — это неизвестная аннотация, а не отрицательный класс; главный критерий — может ли пример без разметки для Task B дать полезный сигнал для Task B, а не просто быть проигнорированным этой head. :chatgpt-content-reference{index="0"}

Поиск охватывал publisher/proceedings pages, IEEE/ACM/ScienceDirect/Springer, CVF, PMLR, AAAI, PubMed/DBLP/OpenAlex, arXiv и code/project pages. Для рейтингов конференций использован ICORE/CORE 2026; в частности, AAAI и ACM MM там A*. :chatgpt-content-reference{index="1"} Для журнальных квартилей я использовал доступные SJR/SCImago-проверки и не приписывал Q1 там, где не смог надежно подтвердить. Прямой массовый экспорт Scopus/WoS/Google Scholar недоступен через используемый интерфейс, поэтому ниже нет выдуманных citation counts; где число цитирований доступно из публичных индексов, это отмечено отдельно.

# 1. Executive summary

1. **Полная четырехкомпонентная постановка — multimodal + multi-task + multi-label + dataset-level missing tasks — остается слабо закрытой.** Я не нашел сильной работы 2025–2026 гг., которая в одном методе решала бы ровно: Dataset A имеет multimodal samples и только Task A multilabel-разметку, Dataset B — те же/другие модальности и только Task B multilabel-разметку, причем модель должна получить прямой обучающий сигнал для обеих heads от обоих датасетов. Современная литература решает отдельные части этой задачи очень хорошо, но не их полное пересечение.

2. Для **S2: dataset-level missing tasks** наиболее прямой современный baseline — **PASAL (Information Fusion, 2025)**. Он позволяет совместно учить heterogeneous datasets с dataset-specific heads и sparse losses. Однако sample из A при отсутствии B-label **не обновляет B-head**: он влияет на B лишь косвенно через shared representation. :chatgpt-content-reference{index="2"}

3. **HiTTs (ACM MM 2025)** — наиболее интересная найденная работа для вашего центрального вопроса «как превратить отсутствующую task annotation в прямой training signal». Вместо одного masking HiTTs учит hierarchical task tokens и через них извлекает pseudo supervision для неразмеченных задач на feature- и prediction-level. Это уже **direct cross-task pseudo-supervision**, а не только shared-backbone transfer. Ограничение: эксперименты относятся к partially annotated dense prediction, а не к multimodal multilabel classification. :chatgpt-content-reference{index="3"}

4. **UNITI (Expert Systems with Applications, 2025)** непосредственно рассматривает inter-dataset MTL на разных датасетах и борется с feature interference / catastrophic forgetting посредством последовательного обучения и feature-level knowledge distillation. Это очень релевантно для стабильности A+B, но само по себе не восстанавливает отсутствующую B-разметку на A. :chatgpt-content-reference{index="4"}

5. В области **incomplete multi-view + missing multi-label learning** 2025–2026 гг. наблюдается заметно более зрелая линия: ICML 2025, CVPR 2025 и несколько TPAMI 2026 работ. Здесь уже системно используются cross-view information, pseudo-labeling, uncertainty, semantic label embeddings и label-aware fusion. :chatgpt-content-reference{index="5"} Но **multi-view ≠ автоматически multimodal**: во многих benchmark’ах views — это несколько feature representations одного объекта, а не, например, raw image+text.

6. Для борьбы с confirmation bias наиболее убедительные современные механизмы — **independent/dual-branch pseudo-labeling**, **category-specific adaptive thresholds**, **evidential uncertainty**, а также использование только надежной части псевдоразметки. ICML 2025 использует dual-branch soft pseudo-label cross-imputation; CTRL/TPAMI 2026 — Beta evidential network + Dempster–Shafer uncertainty; SATL — category-aware thresholds. :chatgpt-content-reference{index="6"}

7. Для **semantic/label transfer** развивается отдельная сильная ветвь VLM/CLIP: HST, PositiveCoOp, VA³P, SCINet. Она особенно интересна в вашем случае, потому что позволяет создавать сигнал для отсутствующей label space на основании semantic descriptions/classes, а не только корреляций, увиденных в том же датасете. :chatgpt-content-reference{index="7"}

8. Практически наиболее содержательная архитектура для вашего A+B — не один опубликованный метод «как есть», а **гибрид из трех проверенных идей**: sparse supervised MTL/PASAL как безопасный базовый слой; teacher/HiTTs-like pseudo-supervision для прямого обновления отсутствующей head; uncertainty/VLM/label-semantic mechanisms для фильтрации и независимой проверки pseudo-labels.

---

# 2. Формализация проблемы

Ваши S1–S6 хорошо отражают реально разделенные в литературе режимы: example-level пропуски, dataset-level missing tasks, disjoint/overlapping label spaces, modality-dependent и mixed supervision. :chatgpt-content-reference{index="8"}

Удобно записать общий случай как

\[
y_{itk}\in\{0,1,?\},
\qquad
m_{itk}=\mathbf 1[y_{itk}\neq ?],
\]

где \(i\) — sample, \(t\) — task/dataset head, \(k\) — label внутри task.

Тогда минимальный корректный supervised loss:

\[
\mathcal L_{\rm sup} =
\frac{
\sum_{i,t,k}
m_{itk}\,
\ell(\hat y_{itk},y_{itk})
}{
\sum_{i,t,k}m_{itk}
}.
\]

Критически важно: для `?` **не вычисляется отрицательный BCE term**.

Однако этот loss отвечает лишь на вопрос «как не повредить обучение ложными negatives». Он **не отвечает** на более важный вопрос:

\[
m_{iBk}=0
\quad\Longrightarrow\quad
\text{может ли sample }i\text{ создать gradient для }h_B?
\]

И здесь литература естественно делится на четыре уровня:

| Уровень | Что происходит с sample A без B-label |
|---|---|
| Ignore | B-loss отсутствует; B-head не получает gradient |
| Representation transfer | sample обновляет shared encoder, B-head — нет |
| Relation/consistency transfer | B получает косвенное constraint через связь с A |
| Direct recovered supervision | для B создается pseudo-target / teacher target / semantic target и B-head обновляется напрямую |

Именно последняя строка ближе всего к вашей центральной цели. :chatgpt-content-reference{index="9"}

---

# 3. Multi-task learning с partial task annotations

## 3.1. HiTTs — наиболее прямой современный механизм восстановления task supervision

**Jingdong Zhang, Hanrong Ye, Xin Li, Wenping Wang, Dan Xu. _Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions_. ACM Multimedia 2025. DOI: 10.1145/3746027.3755727.**

ACM MM 2025 — peer-reviewed; ICORE 2026 относит ACM Multimedia к A*. :chatgpt-content-reference{index="10"}

**Setting:** частичная task annotation для dense tasks — segmentation, depth, normals и т. п. Это ближе к S1/S2, хотя dataset split не полностью совпадает с вашим A/B.

**Architecture:** multi-task backbone + global task tokens + task-specific fine-grained spatial tokens.

**Ключевой механизм:** global tokens взаимодействуют между задачами, fine-grained tokens получают task-specific локальное представление; затем tokens используются для **discovery pseudo task labels на feature- и prediction-level**. Авторы прямо противопоставляют это предыдущим подходам, которые в основном строили cross-task consistency, но не давали direct pixel-wise supervision отсутствующей задаче. :chatgpt-content-reference{index="11"}

**Knowledge transfer:** `cross-task + pseudo-label`, то есть наиболее интересный для вас вариант.

Если для sample отсутствует Task B, B не просто получает изменения backbone. У sample появляется pseudo-B supervision, позволяющий обновлять B-specific prediction path.

**Ограничение:** spatial tokens и dense pseudo maps нельзя перенести на multilabel classification буквально. Но идея хорошо преобразуется в **task tokens + label tokens**, где fine-grained tokens соответствуют уже не пикселям, а классам multilabel ontology.

---

## 3.2. PASAL — сильный baseline для полностью раздельных датасетов

**Thomas Gorges et al. _PASAL: Progress- and sparsity-aware loss balancing for heterogeneous dataset fusion_. Information Fusion 120, 103038, 2025. DOI: 10.1016/j.inffus.2025.103038.** :chatgpt-content-reference{index="12"}

Здесь постановка почти буквально соответствует S2/S3:

- общий shared network;
- dataset-specific subnetworks/heads;
- разные datasets не обязаны иметь одинаковые annotations;
- loss существует только там, где annotation доступна.

ScienceDirect прямо описывает resulting training как **sparse multi-task scenario**, где samples без соответствующей annotation не производят loss для этой head. При этом samples всех datasets обновляют shared model. :chatgpt-content-reference{index="13"}

PASAL добавляет progress- and sparsity-aware loss balancing, потому что обычные GradNorm/uncertainty-based schemes не рассчитаны авторами на такой sparse regime.

**Главный ответ для вашего случая:**

\[
(x_A,y_A,?)
\]

обновляет:

\[
E_{\rm shared}\quad\checkmark,\qquad
h_A\quad\checkmark,\qquad
h_B\quad\mathbf{\times}.
\]

То есть это **representation-level transfer**, но не missing-label recovery.

PASAL очень важен как control experiment: любая ваша более сложная модель должна доказать, что она выигрывает не просто от joint representation learning и грамотного sparse loss balancing.

---

## 3.3. UNITI — heterogeneous datasets + distillation + защита от interference

**Seunghyun Kim, Yeongje Park, Eui Chul Lee. _UNITI: Framework for multi-task learning across datasets to mitigate overfitting_. Expert Systems with Applications, 2025. DOI: 10.1016/j.eswa.2025.127653.** :chatgpt-content-reference{index="14"}

Авторы явно отмечают отличие от классического intra-dataset MTL, где у одного dataset есть несколько labels, и рассматривают задачи из **разных heterogeneous datasets**. Основная проблема — feature interference и catastrophic forgetting.

Механизм:

- dataset/task-specific teachers;
- shared student;
- sequential/inter-dataset learning;
- feature-level knowledge distillation.

В опубликованных экспериментах используются facial age/emotion и Caltech-101; в статье сообщается улучшение до 5.17 процентного пункта относительно стандартного MTL в отдельных экспериментах. :chatgpt-content-reference{index="15"}

**Knowledge transfer:** distillation + shared representation.

**Но:** UNITI не превращает неизвестную B-label на A в явную B target. Поэтому UNITI лучше рассматривать как **stabilization component**, который можно добавить к pseudo-label architecture.

---

## 3.4. Progressive Pseudo Labeling — очень релевантный аналог из multi-dataset detection

**Kai Ye et al. _Progressive Pseudo Labeling for Multi-Dataset Detection Over Unified Label Space_. IEEE Transactions on Multimedia, online 2024 / volume 27, 2025. DOI: 10.1109/TMM.2024.3521841.** :chatgpt-content-reference{index="16"}

Проблема здесь очень близка структурно: разные detection datasets имеют разные label spaces; отсутствие категории в annotation конкретного dataset не означает, что объект этой категории отсутствует.

Используются:

- teacher–student;
- pseudo-labels для недостающих категорий;
- progressive filtering;
- adaptive confidence/entropy treatment;
- unified label space.

Это S2/S3/S4 и пример **direct pseudo-supervision across datasets**.

Ограничение: object detection, а не arbitrary task heads. Но принцип намного ближе к вашему A/B, чем обычный semi-supervised classifier.

---

## 3.5. Важные predecessors

**DiffusionMTL, CVPR 2024** переносит partially annotated multi-task dense prediction в denoising-diffusion formulation и использует multi-task conditioning, чтобы другие задачи помогали восстанавливать отсутствующую supervision. Это сильный непосредственный predecessor HiTTs.  

**Region-aware Distribution Contrast, ECCV 2024** переносит supervision между dense tasks через region-level distribution alignment/contrast.  

**MTPSL, CVPR 2022** — более старый, но фактически стандартный predecessor для cross-task consistency на partially annotated dense tasks; он остается важен именно потому, что современные работы 2024–2025 сравниваются с этой линией.

---

# 4. Multi-label learning с incomplete labels

Здесь необходимо разделять две несовместимые трактовки термина **partial multi-label**.

В релевантной вам трактовке:

\[
y_k\in\{0,1,?\}
\]

— часть positive/negative labels известна, другие labels неизвестны.

Но значительная часть литературы под *partial multi-label learning* понимает **candidate-label set с лишними false-positive labels**. Такие методы, например некоторые disambiguation papers AAAI/IJCAI 2025–2026, я не считаю основными для вашей постановки.

## 4.1. SATL — category-specific pseudo-label thresholds

**Haoxian Ruan et al. _Learning Semantic-Aware Threshold for Multi-Label Image Recognition with Partial Labels_. Expert Systems with Applications, 2026 bibliographic issue; DOI: 10.1016/j.eswa.2025.129216.** :chatgpt-content-reference{index="17"}

Здесь setting корректный: есть **known и unknown labels**, а missing entries восстанавливаются pseudo-labels.

Главный вклад — вместо одного fixed threshold использовать **semantic/category-aware threshold learning**, потому что score distributions разных classes различны.

Для вашего A/B это особенно полезно после добавления teacher_B(A): нельзя применять единый порог вроде `p>0.9` ко всем B-labels. Более реалистично иметь

\[
\tau_k^+,\quad \tau_k^-,
\]

причем они адаптируются по классу.

---

## 4.2. CLS — co-training против confirmation bias

**Gengyu Lyu, Bohang Sun, Xiang Deng, Songhe Feng. _Addressing Multi-Label Learning with Partial Labels: From Sample Selection to Label Selection_. AAAI 2025. DOI: 10.1609/aaai.v39i18.34119.** :chatgpt-content-reference{index="18"}

Важно ограничение: здесь missing labels — в основном **неразмеченные positive labels**, поэтому setting уже вашего общего `{positive, negative, unknown}`.

Метод использует две сети и **Co-Label Selection**. Вместо того чтобы отбрасывать целые uncertain samples, сети помогают друг другу выбирать/удалять ошибочные label assignments. Авторы мотивируют это именно confirmation bias single-network pseudo-labeling. :chatgpt-content-reference{index="19"}

Полезно как механизм reliability, но переносить assumptions о positive-only missingness на ваш общий случай нельзя.

---

## 4.3. LogicMix — augmentation без превращения unknown в negative

**Chak Fong Chong et al. _LogicMix: Sample mixing data augmentation for multi-label image classification with partial labels_. Pattern Recognition 171, 112186, 2026. DOI: 10.1016/j.patcog.2025.112186.** :chatgpt-content-reference{index="20"}

Авторы явно отмечают проблему методов, которые трактуют missing labels как negatives. LogicMix строит augmentation на логике multilabel union, чтобы создавать новые training examples, не полагаясь на искусственную отрицательную разметку.

Это хороший S1 regularizer, но **не создает сильного cross-task signal** в S2.

---

## 4.4. HST — важный predecessor, действительно восстанавливающий unknown labels

**Tianshui Chen et al. _Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels_. International Journal of Computer Vision 132, 6091–6106, 2024. DOI: 10.1007/s11263-024-02127-2.** :chatgpt-content-reference{index="21"}

HST соответствует вашему определению: known labels используются для получения pseudo-labels неизвестных labels. Есть два пути:

- **intra-image semantic transfer:** image-specific label co-occurrence;
- **cross-image transfer:** category-specific prototypes.

Затем known + recovered pseudo-labels совместно обучают classifier. Авторы сообщают улучшения mAP на COCO, Visual Genome и VOC07. :chatgpt-content-reference{index="22"}

Это хороший precedent для идеи, что unknown entries label matrix — не только пропуски, которые следует mask, а потенциально восстанавливаемая supervision.

---

## 4.5. VLM / prompt-based ветвь

### PositiveCoOp, WACV 2025

Метод использует CLIP prior при partial multilabel annotations и показывает важный отрицательный результат: **negative textual prompts могут ухудшать performance**, поскольку VLM pretraining значительно слабее моделирует языковые утверждения об отсутствии класса. PositiveCoOp оставляет VLM-guided positive prompt, а negative representation учит напрямую в feature space. :chatgpt-content-reference{index="23"}

Это непосредственно важно для `? ≠ negative`: нельзя автоматически превращать unknown class в текстовый «no X».

### VA³P, ACM TOMM 2026

**Visual–Label Alignment and Attribute-Aware Prompt** использует pretrained VLM, local visual–label alignment и attribute-specific prompts для COCO/VOC partial-label recognition. :chatgpt-content-reference{index="24"}

### SCINet, TMM 2026

SCINet рассматривает labels как **known correct / known incorrect / unknown**, то есть формализация очень близка вашей. Он использует off-the-shelf multimodal model, text–image semantic alignment, inter-label/inter-instance co-occurrence и semantic augmentation. Исходный preprint появился 8 июля 2025 г.; работа впоследствии получила DOI IEEE TMM `10.1109/TMM.2026.3705189`, поэтому сейчас ее следует считать не arXiv-only, а peer-reviewed TMM publication. :chatgpt-content-reference{index="25"}

---

# 5. Multimodal / multi-view learning с partial supervision

Здесь находится наиболее насыщенная серия 2025–2026 гг., но есть важная методологическая оговорка:

> **multi-view может включать multimodal data, но многие benchmarks строятся из разных visual/feature views, поэтому результаты нельзя автоматически считать доказательством для raw image+text fusion.**

Сам ICML 2025 paper прямо определяет multi-view широко — multi-feature, multi-sequence и multimodal data. :chatgpt-content-reference{index="26"}

## 5.1. ICML 2025: compact semantics + dual-branch soft pseudo-labels

**Jie Wen et al. _Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification_. ICML 2025, PMLR 267:66467–66480.** :chatgpt-content-reference{index="27"}

Два особенно полезных компонента:

1. representation learning стремится сохранить shared task-relevant semantic information между views и подавить redundant task-irrelevant information;
2. missing labels восстанавливаются **dual-branch soft pseudo-label cross-imputation**.

Вторая идея критична: pseudo-label приходит не от той же самой ветви, которая затем сама подтверждает собственное предсказание, а от независимой ветви. Это один из наиболее разумных механизмов против confirmation bias для вашего A↔B transfer.

**Supervision:** S6.  
**Transfer:** cross-view representation + pseudo-label.  
**S2 fully disjoint tasks:** напрямую не решается.

---

## 5.2. CVPR 2025: disentangled view information + label semantic topology

**Xu Yan, Jun Yin, Jie Wen. _Incomplete Multi-View Multi-label Learning via Disentangled Representation and Label Semantic Embedding_. CVPR 2025. DOI: 10.1109/CVPR52734.2025.02861.** :chatgpt-content-reference{index="28"}

Architecture:

- VAE framework;
- cross-view reconstruction → shared/consistent representation;
- mutual-information constraint → separation consistent vs specific factors;
- label semantic embedding;
- preservation of label relevance topology.

Это хороший пример того, что **label completion необязательно должен быть только prediction-thresholding**: structure label space можно встроить непосредственно в latent representation.

Для вашего A/B adaptation нужна единая semantic layer поверх двух ontologies либо bipartite relations между labels A и B.

---

## 5.3. URDF — uncertainty и trustworthy pseudo-labels

**Jie Wen et al. _Partial Multiview Incomplete Multilabel Learning via Uncertainty-Driven Reliable Dynamic Fusion_. TPAMI, online 2025; volume 48(1), 2026, pp.236–250. DOI: 10.1109/TPAMI.2025.3603677.** :chatgpt-content-reference{index="29"}

URDF делает две вещи, обе важны:

- sample-level fusion weights зависят от uncertainty/reliability views;
- unknown labels не просто игнорируются: trustworthy pseudo-labels для них добавляют дополнительную supervision. :chatgpt-content-reference{index="30"}

Это уже близко к нужной philosophy: **uncertainty должна управлять не только feature fusion, но и тем, разрешаем ли missing task target участвовать в loss.**

---

## 5.4. CTRL — evidential pseudo-label reliability

**Yadong Liu et al. _Learning Compact Semantic Information and Reliable Pseudo-Labels for Incomplete Multi-View Multi-Label Classification_. TPAMI 48(7), 7575–7589, 2026. DOI: 10.1109/TPAMI.2026.3665813.** :chatgpt-content-reference{index="31"}

CTRL:

- извлекает high-purity low-redundancy shared representation;
- использует **Beta Evidential Neural Network**;
- комбинирует evidential output с Dempster–Shafer theory;
- получает label-level uncertainty;
- только затем формирует high-reliability pseudo-labels.

Для вашего случая это один из сильнейших механизмов для вопроса «как не разрушить Task B ошибочными B pseudo-labels на Dataset A».

---

## 5.5. DCSI, TPAMI 2026

**_Disentangling Consistent and Specific Information for Double Incomplete Multi-View Multi-Label Classification_**, TPAMI 48(7):7307–7320, DOI `10.1109/TPAMI.2026.3665097`. :chatgpt-content-reference{index="32"}

Работа разделяет consistent и view-specific information и пытается сохранить оба источника вместо чрезмерного сведения всех views к одному shared vector.

Это важно для heterogeneous datasets: forcing слишком сильной invariance может уничтожать dataset-specific signal.

---

## 5.6. V2L — самый свежий результат на cutoff date

**Chengliang Liu et al. _When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling: A Unified Framework for Incomplete Multi-View Multi-Label Learning_. TPAMI, ahead of print, 31 August 2026. DOI: 10.1109/TPAMI.2026.3728832.** :chatgpt-content-reference{index="33"}

V2L объединяет:

- semantically consistent variational encoding;
- preservation of view-specific evidence;
- **instance-wise and label-wise view relevance**;
- hybrid representation/decision fusion.

То есть вместо вопроса «какая modality надежнее в среднем?» задается более точный:

\[
w_{i,v,k}
=
\text{relevance of view }v
\text{ for label }k
\text{ on instance }i.
\]

Это очень привлекательный механизм для настоящего image+text multilabel setup: некоторый label может опираться преимущественно на text, другой — на image.

Так как paper появился буквально за несколько недель до cutoff, делать вывод о citation influence пока рано. ArXiv version от 7 сентября 2026 г., но peer-reviewed TPAMI status уже подтвержден; это не следует маркировать как arXiv-only preprint. :chatgpt-content-reference{index="34"}

---

## 5.7. TACVI-Net и CDSA

**TACVI-Net**, Neural Networks 187, 107349, 2025, использует task-relevant representation и cross-view autoencoder imputation для восстановления missing views. :chatgpt-content-reference{index="35"}

**CDSA**, Knowledge-Based Systems 318, 113507, 2025, вводит dynamic confidence weighting и label-informed semantic alignment. :chatgpt-content-reference{index="36"}

Оба полезны как multimodal/view fusion machinery, но проблема **missing views** у них существеннее, чем dataset-level missing task annotations, поэтому для вашего core research question это вспомогательные работы.

---

# 6. Методы, объединяющие направления

Вот главный итог поиска по пересечениям.

| Семейство | MTL | Multi-dataset | Multimodal/view | Multi-label | Missing supervision | Прямой сигнал отсутствующей head? |
|---|---:|---:|---:|---:|---:|---|
| PASAL | ✓ | ✓ | architecture-agnostic | возможен | S2/S3 | **Нет** |
| UNITI | ✓ | ✓ | нет фокуса | нет фокуса | S2 | через KD, но не missing target |
| HiTTs | ✓ | — | vision | не MLC | partial tasks | **Да** |
| PPL | multi-dataset | ✓ | vision | multi-class ontology | S2/S3/S4 | **Да, pseudo-label** |
| ICML’25 incomplete MvML | — | — | multi-view | ✓ | S1/S6 | **Да, unknown labels** |
| URDF/CTRL | — | — | multi-view | ✓ | S1/S6 | **Да, reliable pseudo-label** |
| SCINet/VLM line | — | — | VLM prior | ✓ | S1 | **Да, semantic** |

Поэтому самая точная формулировка state of the art такая:

> **Механизмы, необходимые для решения вашей полной задачи, уже существуют, но разбросаны по разным литературным линиям. Единого зрелого general framework для S2 multimodal multi-task multilabel heterogeneous supervision я не обнаружил.**

Это и есть центральный research gap.

---

# 7. Сравнительная таблица основных статей

| Paper | Year / venue | Quality status | MTL | Multi-modal/view | Multi-label | Partial type | Main mechanism | Knowledge transfer |
|---|---|---|---:|---:|---:|---|---|---|
| HiTTs :chatgpt-content-reference{index="37"} | 2025 ACM MM | A* ICORE | ✓ | vision | — | S1/S2-like | task tokens + pseudo task labels | **cross-task + pseudo** |
| PASAL :chatgpt-content-reference{index="38"} | 2025 Information Fusion | Q1 verified in SJR pass | ✓ | agnostic | possible | S2/S3/S6 | sparse heads + progress/sparsity balancing | representation |
| UNITI :chatgpt-content-reference{index="39"} | 2025 ESWA | Q1 SJR :chatgpt-content-reference{index="40"} | ✓ | — | — | S2/S3 | sequential training + feature KD | distillation |
| PPL :chatgpt-content-reference{index="41"} | 2025 TMM | peer-reviewed | multi-dataset | vision | — | S2/S3/S4 | EMA/teacher pseudo-labeling | **pseudo-label** |
| Compact Semantic Info :chatgpt-content-reference{index="42"} | ICML 2025 | top-tier | — | ✓ multi-view | ✓ | S1/S6 | MI + dual-branch pseudo-labels | pseudo + cross-view |
| Disentangled MvML :chatgpt-content-reference{index="43"} | CVPR 2025 | top-tier | — | ✓ | ✓ | S1/S6 | VAE + label-semantic topology | representation + label relations |
| URDF :chatgpt-content-reference{index="44"} | TPAMI 2026 | Q1 | — | ✓ | ✓ | S1/S6 | uncertainty fusion + reliable pseudo-labels | pseudo + cross-view |
| CTRL :chatgpt-content-reference{index="45"} | TPAMI 2026 | Q1 | — | ✓ | ✓ | S1/S6 | evidential uncertainty + pseudo-labels | pseudo |
| DCSI :chatgpt-content-reference{index="46"} | TPAMI 2026 | Q1 | — | ✓ | ✓ | S6 | consistent/specific disentangling | cross-view representation |
| V2L :chatgpt-content-reference{index="47"} | TPAMI ahead-of-print 2026 | Q1 | — | ✓ | ✓ | S6 | instance/label-specific view relevance | cross-view + label-aware |
| SATL :chatgpt-content-reference{index="48"} | ESWA 2026 | Q1 | — | image | ✓ | S1 | class-specific thresholds | pseudo-label |
| CLS :chatgpt-content-reference{index="49"} | AAAI 2025 | A* ICORE | — | image | ✓ | restricted S1 | two-network label selection | pseudo/co-training |
| LogicMix :chatgpt-content-reference{index="50"} | Pattern Recognition 2026 | Q1 :chatgpt-content-reference{index="51"} | — | image | ✓ | S1 | logic-preserving augmentation | indirect |
| HST :chatgpt-content-reference{index="52"} | IJCV 2024 | predecessor | — | image | ✓ | S1 | co-occurrence + prototypes | label relations + pseudo |
| PositiveCoOp :chatgpt-content-reference{index="53"} | WACV 2025 | peer-reviewed | — | VLM prior | ✓ | S1 | positive prompt learning | semantic/VLM |
| VA³P :chatgpt-content-reference{index="54"} | ACM TOMM 2026 | peer-reviewed | — | VLM prior | ✓ | S1 | local visual-label alignment + prompts | semantic/VLM |
| SCINet :chatgpt-content-reference{index="55"} | IEEE TMM 2026 | peer-reviewed | — | VLM prior | ✓ | S1 | semantic co-occurrence + multimodal model | semantic + cross-modal |

### Influential predecessors / preprints

Отдельный поиск **не дал основания включать arXiv-only работу в основной список вместо peer-reviewed alternatives**. Наиболее заметные новые preprints быстро получили formal venue/status:

- **HiTTs:** arXiv `2411.18823`, первая версия 27 Nov 2024 → ACM MM 2025. ACM page сейчас показывает несколько citations; это уже не preprint-only. :chatgpt-content-reference{index="56"}
- **SCINet:** arXiv `2507.05992`, 8 Jul 2025 → IEEE TMM 2026. :chatgpt-content-reference{index="57"}
- **V2L:** arXiv `2609.07525`, 7 Sep 2026, но TPAMI ahead-of-print датирован 31 Aug 2026. :chatgpt-content-reference{index="58"}
- **HST** начинался как более ранний preprint, но с 2024 опубликован в IJCV; Springer сейчас показывает значительное дальнейшее цитирование, поэтому его разумно считать **high-influence predecessor**, а не preprint. :chatgpt-content-reference{index="59"}

Для совсем свежих 2026 работ citation velocity пока статистически малоинформативен; особенно для V2L любые «high influence» claims сейчас были бы преждевременны.

---

# 8. Таксономия современных методов

Более естественное деление литературы выглядит так:

```text
Heterogeneous partial supervision
│
├── A. Route only observed supervision
│   ├── masked / selective loss
│   ├── dataset-specific heads
│   └── sparse loss balancing (PASAL)
│
├── B. Share useful representations
│   ├── shared encoder / multimodal trunk
│   ├── consistent–specific disentanglement
│   ├── cross-view reconstruction
│   └── task-/label-conditioned fusion
│
├── C. Recover missing supervision
│   ├── task-token label discovery (HiTTs)
│   ├── teacher / EMA pseudo-labeling
│   ├── dual-network pseudo-labeling
│   ├── uncertainty/evidential pseudo-labels
│   └── adaptive thresholding
│
├── D. Exploit structured task/label relations
│   ├── cross-task consistency
│   ├── label co-occurrence graphs
│   ├── category prototypes
│   ├── semantic label embeddings
│   └── VLM / prompt-based label priors
│
├── E. Exploit modality/view relationships
│   ├── cross-view reconstruction
│   ├── mutual-information learning
│   ├── confidence-weighted fusion
│   ├── label-dependent view selection
│   └── cross-modal semantic alignment
│
└── F. Stabilize heterogeneous optimization
    ├── sparse/progress-aware loss balancing
    ├── knowledge distillation
    ├── sequential dataset training
    └── uncertainty/reliability weighting
```

Я бы специально отделял **C: recovery of missing supervision** от B: representation sharing. Это различие часто теряется в MTL papers, хотя для вашей задачи оно принципиально.

---

# 9. Детальный анализ механизмов

## Masked loss

Базовый вариант:

\[
\mathcal L_{\rm obs}
=
\sum_{i,t,k}
m_{itk}\lambda_t
\operatorname{BCE}(p_{itk},y_{itk}).
\]

Плюсы: корректен, прост, не создает false negatives.

Минус: если \(m_{iBk}=0\), B-head не получает ничего.

PASAL улучшает **optimization вокруг sparse masking**, но не меняет сам факт отсутствия B target. :chatgpt-content-reference{index="60"}

---

## Shared representation

\[
z_i=F(E_{\rm image}(x_i^I),E_{\rm text}(x_i^T)).
\]

Далее:

\[
p_A=h_A(z_i),\qquad p_B=h_B(z_i).
\]

Dataset A улучшает representation \(z\), и поэтому будущие B predictions потенциально улучшаются.

Это реальный transfer, но **indirect**.

Главная опасность — negative transfer: A и B могут требовать несовместимых features.

PASAL балансирует sparse losses; UNITI атакует feature interference через distillation и sequential training. :chatgpt-content-reference{index="61"}

---

## Pseudo-labeling

Обобщенный вариант:

\[
\tilde y_{iBk}=T_B(x_i),\quad i\in D_A,
\]

\[
q_{iBk}=
\mathbf 1[
\text{confidence}(\tilde y_{iBk})>\tau_{Bk}
].
\]

Loss:

\[
\mathcal L_{\rm pseudo}
=
\sum_{i,t,k}
(1-m_{itk})q_{itk}
\operatorname{BCE}(p_{itk},\tilde y_{itk}).
\]

Тогда sample A уже дает:

\[
\nabla_{\theta_B}\mathcal L_{\rm pseudo}\neq0.
\]

Это качественно другой режим.

HiTTs, PPL, ICML 2025, URDF, CTRL и HST дают разные способы получить и отфильтровать такой signal. :chatgpt-content-reference{index="62"}

---

## Confirmation bias

Наиболее обоснованная комбинация современных идей:

\[
\text{independent teacher}
+
\text{class-specific threshold}
+
\text{uncertainty filter}
+
\text{consistency}.
\]

То есть не:

\[
\tilde y=\mathbf 1[p_{\rm same\ model}>0.5]
\]

с самого начала обучения.

Лучше:

- EMA или separately trained teacher;
- две независимо initialized branches, как в ICML 2025;
- \(\tau_k^+\) и \(\tau_k^-\), как мотивирует SATL;
- evidential uncertainty, как CTRL;
- confidence warm-up;
- consistency across augmentations/modalities;
- периодическая calibration только по truly observed labels. :chatgpt-content-reference{index="63"}


---

## Label relations

HST показывает два практически переносимых механизма:

\[
A_{kl}(x)=P(y_l\mid y_k,x)
\]

для intra-sample label relation и prototypes \(c_k\) для cross-sample transfer. :chatgpt-content-reference{index="64"}

CVPR 2025 предлагает сохранять label relevance topology в semantic embedding space. :chatgpt-content-reference{index="65"}

Для disjoint A/B это можно расширить через label text embeddings:

\[
e_k = E_{\rm text}(\text{description of label }k),
\]

что позволяет задавать relations даже между labels, которые никогда не были совместно размечены.

---

## VLM supervision

VLM дает дополнительный независимый источник:

\[
s_k(x)=
\operatorname{sim}
(E_{\rm vision}(x), E_{\rm text}(\ell_k)).
\]

Это особенно ценно при S2, потому что класс \(B_k\) не обязан иметь co-occurrence statistics внутри Dataset A.

Но PositiveCoOp показывает важную предосторожность: embeddings «отсутствия класса» не обязательно хорошо соответствуют VLM pretraining, поэтому отрицательные pseudo-labels надо генерировать осторожнее положительных. :chatgpt-content-reference{index="66"}

---

# 10. Практическое применение к Dataset A + Dataset B

Ваша целевая постановка из файла — два image+text датасета, один размечен по A, второй по B, возможно с multilabel внутри каждой задачи. :chatgpt-content-reference{index="67"}

Ниже пять стратегий, причем я явно разделяю **прямо поддержанные литературой** и **адаптации известных механизмов**.

## Strategy 1 — PASAL-style sparse multimodal MTL

**Architecture**

\[
z =
F(E_I(x^I),E_T(x^T));
\qquad
p_A=h_A(z),\;p_B=h_B(z).
\]

Separate heads нужны, если label ontologies различны.

**Loss**

\[
L=\lambda_A L_A^{obs}+\lambda_B L_B^{obs},
\]

где коэффициенты балансируются по progress/sparsity в духе PASAL.

**Sample A обновляет:** image encoder, text encoder, fusion, shared trunk, A-head.

**Sample A не обновляет:** B-head.

**Преимущества:** самый безопасный и непосредственно валидированный baseline для heterogeneous datasets.

**Ограничение:** B получает от A только representation transfer. :chatgpt-content-reference{index="68"}

---

## Strategy 2 — HiTTs-inspired task/label-token discovery

Это **адаптация**, поскольку HiTTs валидирован на dense prediction.

Вместо spatial tokens создать:

- global task token \(q_A,q_B\);
- label tokens \(q_{A,k},q_{B,l}\);
- cross-task self-attention;
- multimodal features как ключи/значения.

Для sample A:

\[
q_A\leftrightarrow q_B\leftrightarrow z_A
\]

создает prediction для B и confidence:

\[
\tilde y_B,\;q_B^{conf}.
\]

Высоконадежные B labels становятся pseudo-supervision.

**Sample A теперь обновляет B-head напрямую.**

Это наиболее чистый перенос идеи HiTTs в ваш classification setting. :chatgpt-content-reference{index="69"}

**Риск:** связь A↔B может быть намного слабее, чем geometry relations segmentation/depth/normals. Поэтому task tokens стоит дополнять VLM semantics.

---

## Strategy 3 — Cross-dataset teacher–student label completion

Это, на мой взгляд, наиболее практически обоснованный route.

Сначала:

\[
T_A \leftarrow D_A,\qquad T_B \leftarrow D_B.
\]

Затем:

\[
\tilde y^B_A=T_B(x_A),\qquad
\tilde y^A_B=T_A(x_B).
\]

Unified student обучается на:

\[
\mathcal L =
\mathcal L_{\rm observed}
+
\mu(t)\mathcal L_{\rm pseudo}
+
\gamma\mathcal L_{\rm consistency}
+
\delta\mathcal L_{\rm KD}.
\]

Где

\[
\mathcal L_{\rm pseudo}
=
\sum q_{itk}
\,\mathrm{BCE}
(p_{itk},\tilde y_{itk}).
\]

Здесь следует объединить идеи:

- teacher/EMA: PPL;
- dual independent prediction: ICML 2025;
- label uncertainty: CTRL/URDF;
- category thresholds: SATL;
- feature KD/interference control: UNITI. :chatgpt-content-reference{index="70"}


**Главное:** Dataset A теперь напрямую обучает B-head.

---

## Strategy 4 — VLM/semantic teacher для disjoint label spaces

Для каждого label задать semantic representation:

\[
e_{Ak}=E_{\rm label}(d_{Ak}),\qquad
e_{Bl}=E_{\rm label}(d_{Bl}),
\]

где \(d\) — label name + description.

Multimodal sample representation:

\[
z=F(E_I(x^I),E_T(x^T)).
\]

Pseudo-evidence:

\[
s_k =
\operatorname{sim}(z,e_k).
\]

Дополнительно строится graph:

\[
G_{kl}=
\operatorname{sim}(e_k,e_l)
\]

или learned semantic relation network.

Literature basis:

- HST — semantic relations/prototypes;
- CVPR 2025 — label semantic topology;
- SCINet — off-the-shelf multimodal model + co-occurrence;
- PositiveCoOp / VA³P — VLM prompt alignment. :chatgpt-content-reference{index="71"}


Это особенно перспективно при **почти полностью disjoint annotations**, где empirical A/B co-occurrence невозможно измерить.

**Ограничение:** самый сильный опубликованный evidence сейчас относится к VLM/CLIP, а не к произвольному generative LLM teacher. Поэтому я бы ставил VLM teacher выше свободной LLM-generated annotation.

---

## Strategy 5 — UNITI/PASAL stabilization layer

Эта стратегия не заменяет 3/4, а накладывается поверх них.

Использовать:

- task-specific heads;
- potentially task-specific adapters;
- controlled shared trunk;
- PASAL sparse/progress loss balancing;
- feature distillation от A/B teachers;
- sequential or balanced dataset batches.

Цель — не восстановление labels, а предотвращение того, чтобы сильная A pseudo-training разрушала B representation и наоборот. :chatgpt-content-reference{index="72"}

---

## Какой gradient получает sample A

| Подход | Shared encoder | A-head | B-head |
|---|---:|---:|---:|
| Masked/PASAL | ✓ | ✓ | — |
| Shared representation only | ✓ | ✓ | — |
| UNITI KD | ✓ + KD | ✓ | обычно не direct B target |
| HiTTs-like | ✓ | ✓ | **✓ pseudo** |
| Teacher–student | ✓ | ✓ | **✓ accepted pseudo** |
| VLM semantic teacher | ✓ | ✓ | **✓ semantic/pseudo** |

Вот это, на мой взгляд, самый полезный способ сравнивать методы именно для вашей постановки.

---

## Ответы на ваши 13 практических вопросов

**1. Что применимо напрямую?**  
PASAL — напрямую для sparse heterogeneous datasets; UNITI — для inter-dataset MTL/stability. Для multilabel unknown labels — SATL/CTRL-like reliability mechanisms.

**2. Что требует адаптации?**  
HiTTs, PPL и multi-view incomplete MLC: их missing-supervision machinery переносима, но исходные architecture/task assumptions отличаются.

**3. Что происходит с Dataset A sample при Task B?**  
При masked MTL — B-head не обновляется. При pseudo/semantic strategy — обновляется напрямую.

**4. Можно ли генерировать B pseudo-labels для A?**  
Да; это поддерживается близкими направлениями HiTTs/PPL/HST/CTRL, хотя exact A/B multimodal MTL комбинация остается gap.

**5. Как избегать confirmation bias?**  
Independent teacher/dual branches + uncertainty + per-class thresholds + warm-up + consistency.

**6. Semantic/label relations?**  
Да; HST, CVPR 2025 label semantic embeddings, SCINet.

**7. Multimodal consistency?**  
Да; ICML 2025/URDF/V2L показывают, как использовать complementarity/reliability views, но нужна adaptation с feature-views на actual image+text.

**8. VLM/LLM teacher?**  
Для VLM — да, уже есть peer-reviewed evidence. Для generic LLM-generated labels evidence в exact setting существенно слабее.

**9. Dataset-specific heads?**  
Да, особенно при disjoint ontology. PASAL — сильный precedent.

**10. Shared encoder?**  
Да, но желательно сочетать shared и task-specific capacity.

**11. Negative transfer?**  
UNITI непосредственно ориентирован на interference/catastrophic forgetting; PASAL — на imbalance sparse tasks. :chatgpt-content-reference{index="73"}

**12. Gradient balancing?**  
PASAL — один из наиболее прямо релевантных современных методов.

**13. Почти полностью disjoint annotations?**  
Representation-level решение уже зрелое; direct missing-head supervision существует в соседних задачах, но именно generic multimodal multilabel S2 framework остается открытым.

---

# 11. Research gaps

### Gap 1 — полное пересечение четырех требований

`multimodal + MTL + multilabel + dataset-level missing tasks` существенно менее изучено, чем каждая пара/тройка отдельно.

Особенно мало работ с **genuine independently collected datasets**, а не с искусственным random masking annotations одного исходно полного benchmark.

### Gap 2 — direct supervision при полностью disjoint datasets

PASAL показывает, что такие datasets можно совместно учить, но это representation transfer. HiTTs/PPL показывают, что absent supervision можно восстанавливать, но в более специальных задачах.

Не хватает general method:

\[
D_A:(x,y_A,?)
,\qquad
D_B:(x,?,y_B)
\]

\[
\Downarrow
\]

\[
\nabla_{\theta_B}L(x_A)\neq0,\qquad
\nabla_{\theta_A}L(x_B)\neq0
\]

при контролируемой ошибке pseudo-supervision.

### Gap 3 — masked loss уже не является конечной точкой

Современная литература заменяет «просто mask»:

- task-token label discovery;
- denoising/cross-task consistency;
- teacher–student;
- evidential pseudo-labeling;
- label semantic topology;
- VLM prompts;
- uncertainty-based dynamic fusion.

### Gap 4 — uncertainty и negative transfer изучаются преимущественно раздельно

Одна группа papers решает **trustworthiness pseudo-labels**, другая — **gradient/task interference**.

Очень естественный незакрытый вопрос:

\[
\text{pseudo-label reliability}
\quad\longrightarrow\quad
\text{task gradient weight}.
\]

То есть uncertain pseudo-label должен не только исчезать из BCE, но и менять routing/shared-gradient strength.

### Gap 5 — label semantics особенно нужны при S2

При полностью раздельных datasets невозможно надежно выучить empirical correlation \(P(B|A)\), поскольку A/B никогда не наблюдаются вместе.

Semantic embeddings/VLM позволяют обойти это ограничение, но эта идея еще не интегрирована системно с sparse multi-dataset MTL.

### Gap 6 — modality reliability должна быть label-specific

V2L уже показывает важность instance- и label-wise relevance в multi-view setting. :chatgpt-content-reference{index="74"}

Для image+text A/B следующая естественная постановка:

\[
w_{i,m,t,k}
\]

— reliability modality \(m\) для task \(t\), label \(k\), sample \(i\).

Это намного богаче обычного global fusion weight.

---

# 12. Наиболее перспективные направления для нового исследования

## A. Uncertainty-calibrated cross-task pseudo-supervision

Соединить:

- HiTTs-style direct label discovery;
- PPL teacher–student;
- CTRL/URDF evidential uncertainty;
- SATL class-specific thresholds.

Research hypothesis:

\[
\text{direct pseudo transfer}
+
\text{label-specific uncertainty}
>
\text{masked shared MTL}
\]

для genuine S2 datasets.

Главный gap: существующие работы либо создают direct task supervision, либо хорошо моделируют reliability, но редко делают это одновременно в multi-dataset MTL.

---

## B. Semantic bridge между полностью disjoint task ontologies

Соединить:

- separate A/B heads;
- shared multimodal encoder;
- textual label descriptions;
- VLM label embeddings;
- HST/SCINet-style relation transfer.

Особенно интересен случай:

\[
\mathcal Y_A\cap\mathcal Y_B=\varnothing,
\]

но labels семантически связаны.

Это решает проблему, которую невозможно решить простым empirical co-occurrence graph.

---

## C. Modality-consistent pseudo-label acceptance

Не просто:

\[
p_B>0.9.
\]

А, например:

\[
p_B^{image},p_B^{text},p_B^{fusion}
\]

должны быть согласованы с учетом learned modality reliability.

ICML 2025 dual-branch logic + V2L label-wise view relevance дают сильную основу для такого направления. :chatgpt-content-reference{index="75"}

---

## D. Joint sparse-gradient + pseudo-label reliability optimization

Сейчас PASAL решает sparse loss balancing, а CTRL решает pseudo-label reliability.

Можно объединить:

\[
\lambda_t =
f(
\text{task progress},
\text{sparsity},
\text{pseudo uncertainty}
).
\]

Это позволит напрямую связывать надежность reconstructed supervision с величиной gradient contribution.

---

## E. Новый benchmark protocol

На мой взгляд, это не менее важный research contribution, чем новая architecture.

Нужен benchmark, где:

1. Dataset A и B действительно собраны независимо;
2. A/B annotations полностью или почти полностью disjoint;
3. labels не скрываются искусственно из изначально полного annotation matrix;
4. есть image+text или другие реальные modalities;
5. небольшой **fully annotated audit subset** существует только для evaluation/calibration;
6. оценивается отдельно:
   - observed-task performance;
   - missing-task recovery;
   - calibration;
   - pseudo-label precision;
   - negative transfer;
   - performance на genuinely unseen cross-dataset combinations.

Именно такой protocol позволил бы доказать, что метод решает S2, а не только random missing-label imputation.

---

# 13. Bibliography — проверенные ключевые источники

1. **Zhang, J., Ye, H., Li, X., Wang, W., Xu, D.** *Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions.* ACM Multimedia 2025. DOI `10.1145/3746027.3755727`. :chatgpt-content-reference{index="76"}

2. **Gorges, T. et al.** *PASAL: Progress- and sparsity-aware loss balancing for heterogeneous dataset fusion.* Information Fusion 120, 103038, 2025. DOI `10.1016/j.inffus.2025.103038`. :chatgpt-content-reference{index="77"}

3. **Kim, S., Park, Y., Lee, E.C.** *UNITI: Framework for multi-task learning across datasets to mitigate overfitting.* Expert Systems with Applications, 2025. DOI `10.1016/j.eswa.2025.127653`. :chatgpt-content-reference{index="78"}

4. **Ye, K. et al.** *Progressive Pseudo Labeling for Multi-Dataset Detection Over Unified Label Space.* IEEE Transactions on Multimedia. DOI `10.1109/TMM.2024.3521841`. :chatgpt-content-reference{index="79"}

5. **Wen, J. et al.** *Learning Compact Semantic Information for Incomplete Multi-View Missing Multi-Label Classification.* ICML 2025, PMLR 267:66467–66480. :chatgpt-content-reference{index="80"}

6. **Yan, X., Yin, J., Wen, J.** *Incomplete Multi-View Multi-label Learning via Disentangled Representation and Label Semantic Embedding.* CVPR 2025, 30722–30731. DOI `10.1109/CVPR52734.2025.02861`. :chatgpt-content-reference{index="81"}

7. **Wen, J. et al.** *Partial Multiview Incomplete Multilabel Learning via Uncertainty-Driven Reliable Dynamic Fusion.* TPAMI 48(1):236–250, 2026. DOI `10.1109/TPAMI.2025.3603677`. :chatgpt-content-reference{index="82"}

8. **Liu, Y. et al.** *Learning Compact Semantic Information and Reliable Pseudo-Labels for Incomplete Multi-View Multi-Label Classification.* TPAMI 48(7):7575–7589, 2026. DOI `10.1109/TPAMI.2026.3665813`. :chatgpt-content-reference{index="83"}

9. **Wen, J. et al.** *Disentangling Consistent and Specific Information for Double Incomplete Multi-View Multi-Label Classification.* TPAMI 48(7):7307–7320, 2026. DOI `10.1109/TPAMI.2026.3665097`. :chatgpt-content-reference{index="84"}

10. **Wen, J. et al.** *Multi-Domain Feature Integration Based Trusted Partial Multi-View Incomplete Multi-Label Learning.* TPAMI, online 2026; assigned to volume 48 issue 10. DOI `10.1109/TPAMI.2026.3692653`. Печатный issue датирован октябрем 2026, то есть после текущей даты; сама статья уже доступна online. :chatgpt-content-reference{index="85"}

11. **Liu, C. et al.** *When Semantically Consistent Encoding Meets View-Label Heterogeneity Modeling: A Unified Framework for Incomplete Multi-View Multi-Label Learning.* TPAMI, ahead of print, 2026. DOI `10.1109/TPAMI.2026.3728832`. :chatgpt-content-reference{index="86"}

12. **Zhao, L. et al.** *Task-augmented cross-view imputation network for partial multi-view incomplete multi-label classification.* Neural Networks 187, 107349, 2025. DOI `10.1016/j.neunet.2025.107349`. :chatgpt-content-reference{index="87"}

13. **Chen, J. et al.** *Confidence-Enhanced Dual-Space Semantic Alignment for partial multi-view incomplete multi-label classification.* Knowledge-Based Systems 318, 113507, 2025. DOI `10.1016/j.knosys.2025.113507`. :chatgpt-content-reference{index="88"}

14. **Ruan, H. et al.** *Learning Semantic-Aware Threshold for Multi-Label Image Recognition with Partial Labels.* Expert Systems with Applications. DOI `10.1016/j.eswa.2025.129216`. :chatgpt-content-reference{index="89"}

15. **Lyu, G., Sun, B., Deng, X., Feng, S.** *Addressing Multi-Label Learning with Partial Labels: From Sample Selection to Label Selection.* AAAI 2025, 19251–19259. DOI `10.1609/aaai.v39i18.34119`. :chatgpt-content-reference{index="90"}

16. **Chong, C.F. et al.** *LogicMix: Sample mixing data augmentation for multi-label image classification with partial labels.* Pattern Recognition 171, 112186, 2026. DOI `10.1016/j.patcog.2025.112186`. :chatgpt-content-reference{index="91"}

17. **Chen, T. et al.** *Heterogeneous Semantic Transfer for Multi-label Recognition with Partial Labels.* IJCV 132:6091–6106, 2024. DOI `10.1007/s11263-024-02127-2`. :chatgpt-content-reference{index="92"}

18. **Rawlekar, S., Bhatnagar, S., Ahuja, N.** *PositiveCoOp: Rethinking Prompting Strategies for Multi-Label Recognition with Partial Annotations.* WACV 2025, 5863–5872. DOI `10.1109/WACV61041.2025.00572`. :chatgpt-content-reference{index="93"}

19. **Sun, D. et al.** *Visual–Label Alignment and Attribute-Aware Prompt for Multi-Label Image Recognition with Partial Labels.* ACM TOMM 22(8), Article 215, 2026. DOI `10.1145/3828542`. :chatgpt-content-reference{index="94"}

20. **Wu, X. et al.** *Exploring Partial Multi-Label Learning via Integrating Semantic Co-occurrence Knowledge (SCINet).* IEEE Transactions on Multimedia, 2026. DOI `10.1109/TMM.2026.3705189`; original arXiv `2507.05992`. :chatgpt-content-reference{index="95"}

---

## Итоговая практическая позиция

Если свести обзор к одной экспериментальной программе для вашей постановки, я бы сравнивал **четыре уровня**, а не десятки разрозненных моделей:

\[
\boxed{\text{1. masked sparse MTL}}
\rightarrow
\boxed{\text{2. + calibrated cross-dataset pseudo-labels}}
\rightarrow
\boxed{\text{3. + semantic/VLM label transfer}}
\rightarrow
\boxed{\text{4. + uncertainty + interference-aware weighting}}
\]

Первый уровень показывает benefit общей representation. Второй отвечает на главный научный вопрос — может ли \(x_A\) **непосредственно** обучать Task B. Третий особенно важен при полностью disjoint annotations. Четвертый атакует две основные причины провала такой системы: confirmation bias и negative transfer.

Именно переход от уровня 1 к уровням 2–4 сейчас выглядит наиболее содержательным незакрытым направлением: литература уже дает все необходимые строительные блоки, но пока не объединяет их в general **multimodal, multilabel, multi-task, multi-dataset S2/S6 framework**.