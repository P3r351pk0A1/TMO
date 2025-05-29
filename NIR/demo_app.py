import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

@st.cache_data
def load_data():
    covtype = fetch_covtype(as_frame=True)
    df = covtype.frame
    # Для ускорения работы берём подвыборку
    df = df.sample(5000, random_state=42)
    return df

@st.cache_data
def preprocess_data(df):
    X = df.drop('Cover_Type', axis=1)
    y = df['Cover_Type']
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.7, random_state=1, stratify=y)
    return X_train, X_test, y_train, y_test

models_list = ['Logistic Regression', 'Decision Tree', 'SVM', 'Random Forest', 'Gradient Boosting']
clas_models = {
    'Logistic Regression': LogisticRegression(max_iter=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', max_iter=15, probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=5, max_depth=5, random_state=42)
}

def draw_multiclass_roc(y_test, y_score, classes, ax):
    # Binarize y_test
    y_test_bin = label_binarize(y_test, classes=classes)
    # Micro-average ROC curve and ROC area
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Micro-average ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC-кривая (multi-class, micro-average)')
    ax.legend(loc="lower right")

# --- Массив для хранения гиперпараметров и обученных моделей ---
if 'model_params' not in st.session_state:
    st.session_state.model_params = [None] * len(models_list)
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = [None] * len(models_list)

def get_model_params(model_name):
    # Возвращает гиперпараметры для текущей модели из session_state
    idx = models_list.index(model_name)
    return st.session_state.model_params[idx]

def set_model_params(model_name, params):
    idx = models_list.index(model_name)
    st.session_state.model_params[idx] = params

def set_trained_model(model_name, model):
    idx = models_list.index(model_name)
    st.session_state.trained_models[idx] = model

def get_trained_model(model_name):
    idx = models_list.index(model_name)
    return st.session_state.trained_models[idx]

# --- Гиперпараметры для каждой модели через sidebar ---
def get_model_with_params(model_name, model_idx=0, render_ui=True):
    params = {}
    # Уникальный префикс для ключей с учётом индекса
    key_prefix = f"{model_name.replace(' ', '_').lower()}_{model_idx}"
    # Если не нужно отрисовывать UI, просто вернуть модель с текущими параметрами
    if not render_ui:
        prev_params = get_model_params(model_name)
        if prev_params is not None:
            # Если параметры уже есть в session_state
            if model_name == 'Logistic Regression':
                return LogisticRegression(**prev_params), prev_params
            elif model_name == 'Decision Tree':
                return DecisionTreeClassifier(**prev_params), prev_params
            elif model_name == 'SVM':
                return SVC(**prev_params), prev_params
            elif model_name == 'Random Forest':
                return RandomForestClassifier(**prev_params), prev_params
            elif model_name == 'Gradient Boosting':
                return GradientBoostingClassifier(**prev_params), prev_params
        # Если параметров нет, вернуть дефолтную модель
        return clas_models[model_name], {}
    if model_name == 'Logistic Regression':
        with st.sidebar.expander('Logistic Regression', expanded=True):
            C = st.slider('C (регуляризация)', 0.01, 5.0, 1.0, 0.01, key=f'{key_prefix}_c')
            max_iter = st.slider('max_iter', 50, 500, 100, 10, key=f'{key_prefix}_max_iter')
            params = dict(C=C, max_iter=max_iter, random_state=42)
        return LogisticRegression(**params), params
    elif model_name == 'Decision Tree':
        with st.sidebar.expander('Decision Tree', expanded=True):
            max_depth = st.slider('max_depth', 1, 30, 10, 1, key=f'{key_prefix}_max_depth')
            min_samples_split = st.slider('min_samples_split', 2, 20, 2, 1, key=f'{key_prefix}_min_samples_split')
            params = dict(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        return DecisionTreeClassifier(**params), params
    elif model_name == 'SVM':
        with st.sidebar.expander('SVM', expanded=True):
            C = st.slider('C (регуляризация)', 0.01, 5.0, 1.0, 0.01, key=f'{key_prefix}_c')
            kernel = st.selectbox('kernel', ['rbf', 'linear', 'poly', 'sigmoid'], index=0, key=f'{key_prefix}_kernel')
            max_iter = st.slider('max_iter', 10, 500, 15, 5, key=f'{key_prefix}_max_iter')
            params = dict(C=C, kernel=kernel, max_iter=max_iter, probability=True, random_state=42)
        return SVC(**params), params
    elif model_name == 'Random Forest':
        with st.sidebar.expander('Random Forest', expanded=True):
            n_estimators = st.slider('n_estimators', 5, 200, 5, 5, key=f'{key_prefix}_n')
            max_depth = st.slider('max_depth', 1, 30, 10, 1, key=f'{key_prefix}_max_depth')
            params = dict(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        return RandomForestClassifier(**params), params
    elif model_name == 'Gradient Boosting':
        with st.sidebar.expander('Gradient Boosting', expanded=True):
            n_estimators = st.slider('n_estimators', 5, 200, 5, 5, key=f'{key_prefix}_n')
            learning_rate = st.slider('learning_rate', 0.01, 1.0, 0.1, 0.01, key=f'{key_prefix}_lr')
            max_depth = st.slider('max_depth', 1, 10, 5, 1, key=f'{key_prefix}_max_depth')
            params = dict(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42)
        return GradientBoostingClassifier(**params), params
    else:
        return clas_models[model_name], {}

def show_model_block(model_name, X_train, X_test, y_train, y_test, model_idx=0):
    model, params = get_model_with_params(model_name, model_idx, render_ui=True)
    prev_params = get_model_params(model_name)
    trained_model = get_trained_model(model_name)
    # Проверяем, изменились ли параметры
    if prev_params != params or trained_model is None:
        model.fit(X_train, y_train)
        set_model_params(model_name, params)
        set_trained_model(model_name, model)
    else:
        model = trained_model
    Y_pred = model.predict(X_test)
    Y_pred_proba = model.predict_proba(X_test)
    classes = model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)
    roc_auc = roc_auc_score(y_test_bin, Y_pred_proba, average='micro', multi_class='ovr')
    fig, ax = plt.subplots(ncols=2, figsize=(12,5))
    draw_multiclass_roc(y_test, Y_pred_proba, classes, ax[0])
    cm = confusion_matrix(y_test, Y_pred, normalize='true', labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax[1], cmap=plt.cm.Blues, colorbar=False)
    ax[1].set_title('Матрица ошибок')
    fig.suptitle(f"{model_name} (ROC-AUC: {roc_auc:.3f})")
    st.pyplot(fig)

# --- Блок ручного ввода признаков и предсказания ---
def manual_predict_block(X_train, scaler, model, feature_names):
    st.sidebar.markdown('---')
    st.sidebar.header('Ввод признаков для прогноза')
    st.sidebar.markdown('Введите значения признаков для одного участка:')
    input_data = {}
    # 10 непрерывных признаков
    cont_features = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]
    rus_cont_features = [
        'Высота (Elevation)', 'Ориентация склона (Aspect)', 'Уклон (Slope)', 'Горизонтальная дистанция до воды',
        'Вертикальная дистанция до воды', 'Горизонтальная дистанция до дороги',
        'Освещённость в 9:00', 'Освещённость в полдень', 'Освещённость в 15:00', 'Горизонтальная дистанция до пожарных пунктов'
    ]
    for feat, rus_feat in zip(cont_features, rus_cont_features):
        min_val = float(X_train[feat].min())
        max_val = float(X_train[feat].max())
        val = st.sidebar.number_input(f'{rus_feat}', min_value=min_val, max_value=max_val, value=float(X_train[feat].mean()), key=feat)
        input_data[feat] = val
    # Wilderness Area (один выбор из 4)
    wilderness_options = [f'Заповедник {i}' for i in range(1, 5)]
    wilderness_selected = st.sidebar.selectbox('Заповедник (Wilderness Area)', wilderness_options, index=0)
    for i in range(1, 5):
        input_data[f'Wilderness_Area{i}'] = 1 if wilderness_selected == f'Заповедник {i}' else 0
    # Soil Type (один выбор из 40)
    soil_types = [f'Тип почвы {i}' for i in range(1, 41)]
    selected_soil = st.sidebar.selectbox('Тип почвы (Soil Type)', soil_types, index=0)
    for idx, stype in enumerate(soil_types, 1):
        input_data[f'Soil_Type{idx}'] = 1 if selected_soil == stype else 0
    # Собираем в DataFrame
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Масштабируем как обучающие данные
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
    if st.sidebar.button('Сделать прогноз'):
        pred = model.predict(input_scaled)[0]
        st.sidebar.success(f'Прогноз типа лесного покрова (Cover_Type): {pred}')

st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list, default=['Random Forest'])

# --- Кнопка и слайдер для изменения размера подвыборки ---
st.sidebar.markdown('---')
st.sidebar.header('Настройки данных')
sample_size = st.sidebar.slider('Размер подвыборки для обучения', min_value=1000, max_value=100000, value=5000, step=500, key='sample_size_slider')
if 'current_sample_size' not in st.session_state or st.session_state.current_sample_size != sample_size:
    st.session_state.current_sample_size = sample_size
    # Сбросить обученные модели и параметры при изменении размера подвыборки
    st.session_state.model_params = [None] * len(models_list)
    st.session_state.trained_models = [None] * len(models_list)

# --- Загрузка данных с учетом выбранного размера подвыборки ---
@st.cache_data(hash_funcs={pd.DataFrame: lambda _: None}, show_spinner=False)
def load_data_custom(sample_size):
    covtype = fetch_covtype(as_frame=True)
    df = covtype.frame
    df = df.sample(sample_size, random_state=42)
    return df

df = load_data_custom(sample_size)
X_train, X_test, y_train, y_test = preprocess_data(df)

# Получаем scaler, обученный на X_train
scaler = MinMaxScaler().fit(X_train)
feature_names = X_train.columns.tolist()

st.header('Оценка качества моделей (многоклассовая классификация)')

# --- Блок ручного прогноза для всех выбранных моделей ---
if models_select:
    # Блок ручного ввода признаков (один раз)
    st.sidebar.markdown('---')
    st.sidebar.header('Ввод признаков для прогноза')
    st.sidebar.markdown('Введите значения признаков для одного участка:')
    input_data = {}
    # Абсолютные значения признаков (НЕ масштабированные)
    # Получаем min/max/mean по НЕмасштабированным данным
    df_full = load_data()
    cont_features = [
        'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
    ]
    rus_cont_features = [
        'Высота (Elevation), м',
        'Ориентация склона (Aspect), градусы',
        'Уклон (Slope), градусы',
        'Горизонтальная дистанция до воды, м',
        'Вертикальная дистанция до воды, м',
        'Горизонтальная дистанция до дороги, м',
        'Освещённость в 9:00 (Hillshade_9am), 0-255',
        'Освещённость в полдень (Hillshade_Noon), 0-255',
        'Освещённость в 15:00 (Hillshade_3pm), 0-255',
        'Горизонтальная дистанция до пожарных пунктов, м'
    ]
    for feat, rus_feat in zip(cont_features, rus_cont_features):
        min_val = float(df_full[feat].min())
        max_val = float(df_full[feat].max())
        mean_val = float(df_full[feat].mean())
        val = st.sidebar.number_input(f'{rus_feat}', min_value=min_val, max_value=max_val, value=mean_val, key=f'manual_{feat}')
        input_data[feat] = val
    wilderness_options = [f'Заповедник {i}' for i in range(1, 5)]
    wilderness_selected = st.sidebar.selectbox('Заповедник (Wilderness Area)', wilderness_options, index=0, key='manual_wilderness')
    for i in range(1, 5):
        input_data[f'Wilderness_Area{i}'] = 1 if wilderness_selected == f'Заповедник {i}' else 0
    soil_types = [f'Тип почвы {i}' for i in range(1, 41)]
    selected_soil = st.sidebar.selectbox('Тип почвы (Soil Type)', soil_types, index=0, key='manual_soil')
    for idx, stype in enumerate(soil_types, 1):
        input_data[f'Soil_Type{idx}'] = 1 if selected_soil == stype else 0
    # Собираем DataFrame с абсолютными значениями, затем МАСШТАБИРУЕМ как обучающие данные
    input_df = pd.DataFrame([input_data], columns=feature_names)
    # Проверка на пропуски и заполнение нулями (или другим значением)
    if input_df.isnull().any().any():
        input_df = input_df.fillna(0)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_names)
    if st.sidebar.button('Сделать прогноз', key='manual_predict_btn'):
        st.sidebar.markdown('### Результаты прогноза:')
        for idx, model_name in enumerate(models_select):
            model, _ = get_model_with_params(model_name, model_idx=idx, render_ui=False)
            prev_params = get_model_params(model_name)
            trained_model = get_trained_model(model_name)
            if prev_params != _ or trained_model is None:
                model.fit(X_train, y_train)
                set_model_params(model_name, _)
                set_trained_model(model_name, model)
            else:
                model = trained_model
            try:
                pred = model.predict(input_scaled)[0]
                st.sidebar.success(f'{model_name}: {pred}')
            except Exception as e:
                st.sidebar.error(f'{model_name}: ошибка прогноза — {e}')

for idx, model_name in enumerate(models_select):
    with st.expander(f'Настройки и результаты: {model_name}', expanded=True):
        show_model_block(model_name, X_train, X_test, y_train, y_test, idx)

st.markdown('---')
st.write('Изменяйте параметры моделей в соответствующих блоках и анализируйте ROC-кривые и матрицы ошибок!')
