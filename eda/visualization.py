import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve
)
import plotly.express as px
import plotly.graph_objects as go


def plot_confusion_matrix(y_test, y_pred):
    # Рассчет матрицы ошибок
    cm = confusion_matrix(y_test.to_numpy(), y_pred.to_numpy())
    tn, fp, fn, tp = cm.ravel()

    # Рассчет долей
    total = cm.sum()
    percentages = (cm / total * 100).round(1)

    # Создаем красивую палитру (пастельные тона)
    colorscale = [
        [0.0, '#F8E2E2'],   # Светло-красный для TP
        [0.5, '#F8E2E2'],  # Светло-красный для FP/FN
        [1.0, '#E0F2E9'],  # Светло-зеленый для TN
    ]

    # Создаем текст для ячеек с переносом строк
    annotations = []
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            perc = percentages[i, j]
            annotations.append(
                f"<b>{value}</b><br>({perc}%)"
            )

    annotations = np.array(annotations).reshape(2, 2)

    # Создаем фигуру
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Legit', 'Fraud'],
        y=['Legit', 'Fraud'],
        colorscale=colorscale,
        text=annotations,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverinfo='none',
        showscale=False,
        xgap=2,
        ygap=2
    ))

    # Настраиваем аннотации для каждого квадранта
    quadrant_annotations = [
        dict(
            x=0.25, y=0.25,
            xref="x", yref="y",
            text="<b>TN</b><br>True Negative",
            showarrow=False,
            font=dict(size=14, color="#0d5c31")
        ),
        dict(
            x=0.75, y=0.25,
            xref="x", yref="y",
            text="<b>FP</b><br>False Positive",
            showarrow=False,
            font=dict(size=14, color="#9e2a2a")
        ),
        dict(
            x=0.25, y=0.75,
            xref="x", yref="y",
            text="<b>FN</b><br>False Negative",
            showarrow=False,
            font=dict(size=14, color="#9e2a2a")
        ),
        dict(
            x=0.75, y=0.75,
            xref="x", yref="y",
            text="<b>TP</b><br>True Positive",
            showarrow=False,
            font=dict(size=14, color="#0d5c31")
        )
    ]

    # Настраиваем внешний вид
    fig.update_layout(
        title='<b>Матрица ошибок</b>',
        title_x=0.52,
        title_font=dict(size=20),
        xaxis=dict(
            tickfont=dict(size=14),
            side='top'
        ),
        yaxis=dict(
            tickfont=dict(size=14),
            autorange='reversed',
            tickangle=-90  # Вертикальные подписи
        ),
        height=600,
        width=700,
        annotations=quadrant_annotations,
        margin=dict(l=80, r=50, t=100, b=80),
        plot_bgcolor='white'
    )

    # Добавляем подписи значений
    fig.add_annotation(
        x=-0.15, y=0.5,
        xref="paper", yref="y",
        text="<b>Фактические<br>значения</b>",
        showarrow=False,
        font=dict(size=16),
        textangle=-90
    )

    fig.add_annotation(
        x=0.5, y=1.1,
        xref="paper", yref="paper",
        text="<b>Предсказанные значения</b>",
        showarrow=False,
        font=dict(size=16)
    )

    correct_color = '#2E7D32'  # Зеленый
    error_color = '#D32F2F'    # Красный
    text_color = 'black'

    # Цветовая легенда
    fig.add_annotation(
        x=0.5, y=-0.10,
        xref="paper", yref="paper",
        text=f"<span style='color:{correct_color};font-weight:600'>Correct Predictions</span> | " +
             f"<span style='color:{error_color};font-weight:600'>Errors</span>",
        showarrow=False,
        font=dict(size=14, color=text_color)
    )

    # Добавим важные метрики
    metrics_text = (
        f"<b>Accuracy</b>: {(tp + tn)/total:.2%} | "
        f"<b>Recall</b>: {tp/(tp + fn):.2%} | "
        f"<b>Precision</b>: {tp/(tp + fp):.2%}"
    )

    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=12, color=text_color),
        bgcolor="rgba(240,240,240,0.5)",
        bordercolor=text_color,
        borderwidth=1
    )

    return fig


def plot_roc_auc(y_true, y_scores, title='ROC Curve', theme='dark'):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    # Находим оптимальную точку
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    # Выбираем цветовую схему
    if theme == 'dark':
        bg_color = '#121212'
        text_color = 'white'
        grid_color = 'rgba(255, 255, 255, 0.1)'
        fill_color = 'rgba(76, 175, 80, 0.3)'
        line_color = '#4CAF50'
        baseline_color = '#F44336'
        optimal_color = '#FFC107'
        diagonal_color = '#F44336'
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        fill_color = 'rgba(76, 175, 80, 0.2)'
        line_color = '#2E7D32'
        baseline_color = '#D32F2F'
        optimal_color = '#FF8F00'
        diagonal_color = '#B71C1C'

    # Создаем фигуру
    fig = go.Figure()

    # Основная кривая ROC
    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.4f})',
        line=dict(color=line_color, width=3),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>FPR</b>: %{x:.3f}<br><b>TPR</b>: %{y:.3f}<br><b>Threshold</b>: %{text:.4f}<extra></extra>',
        text=thresholds
    ))

    # Диагональ (случайный классификатор)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color=diagonal_color, width=2, dash='dash'),
        hovertemplate='Random: TPR = FPR<extra></extra>'
    ))

    # Оптимальная точка
    fig.add_trace(go.Scatter(
        x=[optimal_fpr],
        y=[optimal_tpr],
        mode='markers+text',
        name=f'Optimal Threshold ({optimal_threshold:.2f})',
        marker=dict(color=optimal_color, size=12),
        text=[f'Threshold: {optimal_threshold:.2f}<br>TPR: {optimal_tpr:.3f}<br>FPR: {optimal_fpr:.3f}'],
        textposition='top right',
        hovertemplate='<b>Optimal Point</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<br>%{text}<extra></extra>'
    ))

    # Линия к оптимальной точке
    fig.add_shape(
        type="line",
        x0=optimal_fpr,
        y0=0,
        x1=optimal_fpr,
        y1=optimal_tpr,
        line=dict(color=optimal_color, width=1, dash="dot")
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=optimal_tpr,
        x1=optimal_fpr,
        y1=optimal_tpr,
        line=dict(color=optimal_color, width=1, dash="dot")
    )

    # Настройка макета
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:14px">AUC: {roc_auc:.4f} | Optimal Threshold: {optimal_threshold:.4f}</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=text_color)
        ),
        xaxis=dict(
            title='False Positive Rate (FPR)',
            range=[0, 1.01],
            gridcolor=grid_color,
            tickformat='.1f',
            title_font=dict(size=16, color=text_color)
        ),
        yaxis=dict(
            title='True Positive Rate (TPR)',
            range=[0, 1.01],
            gridcolor=grid_color,
            tickformat='.1f',
            title_font=dict(size=16, color=text_color)
        ),
        legend=dict(
            x=0.05,
            y=0.05,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color)),
        hovermode='x unified',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=600,
        width=800,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family="Arial", color=text_color)
    )

    # Добавляем аннотацию с информацией
    fig.add_annotation(
        x=optimal_fpr,
        y=optimal_tpr,
        text=f"Threshold: {optimal_threshold:.2f}<br>J-index: {j_scores[optimal_idx]:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=-40 if optimal_fpr > 0.3 else 40,
        ay=-40 if optimal_tpr > 0.7 else 40,
        font=dict(size=12, color=optimal_color),
        bgcolor="rgba(30,30,30,0.7)" if theme == 'dark' else "rgba(240,240,240,0.7)",
        bordercolor=optimal_color,
        borderwidth=1,
        borderpad=4
    )


    # Линия от (0,0) до оптимальной точки и до (1,1)
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color=diagonal_color, width=1, dash="dash")
    )

    return fig


def plot_pr_auc(y_true, y_scores, title='Precision-Recall Curve', theme='dark'):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    # Находим индекс оптимальной точки (максимальное F1-score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]

    # Создаем базовую линию (случайный классификатор)
    baseline = np.sum(y_true) / len(y_true)

    # Выбираем цветовую схему
    if theme == 'dark':
        bg_color = '#121212'
        text_color = 'white'
        grid_color = 'rgba(255, 255, 255, 0.1)'
        fill_color = 'rgba(76, 175, 80, 0.3)'
        line_color = '#4CAF50'
        baseline_color = '#F44336'
        optimal_color = '#FFC107'
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        fill_color = 'rgba(76, 175, 80, 0.2)'
        line_color = '#2E7D32'
        baseline_color = '#D32F2F'
        optimal_color = '#FF8F00'

    # Создаем фигуру
    fig = go.Figure()

    # Основная кривая PR
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.4f})',
        line=dict(color=line_color, width=3),
        fill='tozeroy',
        fillcolor=fill_color,
        hovertemplate='<b>Recall</b>: %{x:.3f}<br><b>Precision</b>: %{y:.3f}<extra></extra>'
    ))

    # Линия базового классификатора
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[baseline, baseline],
        mode='lines',
        name=f'Baseline (AP = {baseline:.4f})',
        line=dict(color=baseline_color, width=2, dash='dash'),
        hovertemplate='Baseline: %{y:.3f}<extra></extra>'
    ))

    # Адаптивное позиционирование для оптимальной точки
    text_position = 'bottom left' if recall[optimal_idx] > 0.7 else 'top right'

    # Оптимальная точка
    fig.add_trace(go.Scatter(
        x=[recall[optimal_idx]],
        y=[precision[optimal_idx]],
        mode='markers',
        name=f'Optimal Threshold ({optimal_threshold:.2f})',
        marker=dict(color=optimal_color, size=12),
        hovertemplate=f'<b>Optimal Point</b><br>Recall: {recall[optimal_idx]:.3f}<br>Precision: {precision[optimal_idx]:.3f}<br>Threshold: {optimal_threshold:.2f}<br>F1: {f1_scores[optimal_idx]:.3f}<extra></extra>'
    ))

    # Текст рядом с точкой (отдельный элемент)
    fig.add_annotation(
        x=recall[optimal_idx],
        y=precision[optimal_idx],
        text=f"Threshold: {optimal_threshold:.2f}<br>F1: {f1_scores[optimal_idx]:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=-40 if recall[optimal_idx] > 0.7 else 40,
        ay=-40 if precision[optimal_idx] > 0.7 else 40,
        font=dict(size=12, color=optimal_color),
        bgcolor="rgba(30,30,30,0.7)" if theme == 'dark' else "rgba(240,240,240,0.7)",
        bordercolor=optimal_color,
        borderwidth=1,
        borderpad=4
    )

    # Настройка макета
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><span style="font-size:14px">Average Precision: {pr_auc:.4f} | Baseline: {baseline:.4f}</span>',
            x=0.5,
            xanchor='center',
            font=dict(size=20, color=text_color)
        ),
        xaxis=dict(
            title='Recall',
            range=[0, 1.01],
            gridcolor=grid_color,
            tickformat='.1f',
            title_font=dict(size=16, color=text_color)
        ),
        yaxis=dict(
            title='Precision',
            range=[0, 1.01],
            gridcolor=grid_color,
            tickformat='.1f',
            title_font=dict(size=16, color=text_color)
        ),
        legend=dict(
            x=0.05,
            y=0.05,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color=text_color)
        ),
        hovermode='x unified',
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        height=600,
        width=800,
        margin=dict(l=80, r=50, t=100, b=80),
        font=dict(family="Arial", color=text_color)
    )

    return fig


def plot_shap(shap_matrix, feature_names, title='Global Feature Importance', theme='dark', top_k=20):
    # Рассчет глобальной важности (среднее абсолютное значение SHAP)
    global_shap = np.abs(shap_matrix).mean(axis=0)

    # Создаем DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': global_shap
    }).sort_values('SHAP_Value', ascending=False).head(top_k)

    # Сортировка для правильного отображения
    importance_df = importance_df.sort_values('SHAP_Value', ascending=True)

    # Цветовые схемы
    if theme == 'dark':
        bg_color = '#121212'
        text_color = 'white'
        grid_color = 'rgba(255, 255, 255, 0.1)'
        color_scale = px.colors.sequential.Blues
        bar_color = '#64b5f6'
        hover_bg = 'rgba(30,30,30,0.7)'
    else:
        bg_color = 'white'
        text_color = 'black'
        grid_color = 'rgba(0, 0, 0, 0.1)'
        color_scale = px.colors.sequential.Blues
        bar_color = '#1976d2'
        hover_bg = 'rgba(240,240,240,0.7)'

    # Создаем фигуру
    fig = px.bar(
        importance_df,
        x='SHAP_Value',
        y='Feature',
        orientation='h',
        title=f'<b>{title}</b><br><span style="font-size:14px">Top {top_k} Features by Mean |SHAP| Value</span>',
        color='SHAP_Value',
        color_continuous_scale=color_scale,
        text='SHAP_Value',
        hover_data={'Feature': True, 'SHAP_Value': ':.4f'},
        height=600 + (top_k - 20) * 10  # Динамическая высота
    )

    # Настраиваем оформление
    fig.update_traces(
        texttemplate='%{text:.4f}',
        textposition='outside',
        marker_color=bar_color,
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>',
        textfont=dict(color=text_color)
    )

    fig.update_layout(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(family="Arial", color=text_color, size=12),
        xaxis=dict(
            title='Mean |SHAP Value|',
            gridcolor=grid_color,
            title_font=dict(size=14, color=text_color)
        ),
        yaxis=dict(
            title=None,
            tickfont=dict(size=12, color=text_color)
        ),
        title=dict(
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=20, color=text_color)
        ),
        margin=dict(l=120, r=50, t=100, b=80),
        hoverlabel=dict(
            bgcolor=hover_bg,
            font_size=12,
            font_family="Arial"
        ),
        showlegend=False,
        coloraxis_showscale=False
    )

    # Добавляем аннотации для контекста
    fig.add_annotation(
        x=0.98,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Higher value → More important",
        showarrow=False,
        font=dict(size=16, color=text_color),
        bgcolor="rgba(30,30,30,0.5)" if theme == 'dark' else "rgba(200,200,200,0.5)",
        bordercolor=text_color,
        borderwidth=1,
        borderpad=2
    )

    # Добавляем линию среднего значения
    mean_shap = importance_df['SHAP_Value'].mean()
    fig.add_shape(
        type="line",
        x0=mean_shap,
        y0=-0.5,
        x1=mean_shap,
        y1=top_k-0.5,
        line=dict(
            color=text_color,
            width=1,
            dash="dot"
        )
    )
    fig.add_annotation(
        x=mean_shap,
        y=top_k-0.5,
        text=f" Mean: {mean_shap:.4f}",
        showarrow=False,
        yshift=10,
        font=dict(size=12, color=text_color)
    )

    return fig