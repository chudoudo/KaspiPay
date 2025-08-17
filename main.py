import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

df = pd.read_excel("Бизнес-кейс для кандидата KP.xlsx", sheet_name = "Данные")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Парсинг

def parse_period(col: str) -> pd.Period:
    date_str = col.split("_")[1]  # "01DEC2024"
    dt = datetime.strptime(date_str, "%d%b%Y")
    return pd.Period(dt.strftime("%Y-%m"), freq="M")

def make_col(prefix: str, period: pd.Period) -> str:
    dt = period.to_timestamp(how="S")
    return f"{prefix}_{dt.strftime('%d%b%Y').upper()}"

# Метрики

def compute_engagement(df, month):
    prev3 = [month - i for i in [1, 2, 3]]
    prev3_cols = [make_col("TR", m) for m in prev3]
    if not all(c in df for c in prev3_cols):
        return pd.Series([0]*len(df), index=df.index), prev3
    engaged = (df[prev3_cols] > 0).all(axis=1).astype(int)
    return engaged, prev3

def monthly_metrics(df, month, months):
    tr_col = make_col("TR", month)
    amt_col = make_col("AMT", month)
    flg_col = make_col("FLG", month)

    if tr_col not in df or amt_col not in df:
        return None

    engaged, prev3 = compute_engagement(df, month)

    # Активные клиенты в этом месяце
    buyers = (df[tr_col] > 0) if tr_col in df else (df[flg_col] == 1)

    # Новые клиенты = не было покупок раньше
    prior_tr_cols = [make_col("TR", pm) for pm in months if pm < month and make_col("TR", pm) in df]
    new_mask = (df[prior_tr_cols].fillna(0).sum(axis=1) == 0) & buyers

    # Вовлечённые
    engaged_mask = (engaged == 1) & buyers

    # Остальные
    other_mask = buyers & ~new_mask & ~engaged_mask

    # Дополнительные метрики
    total_amount = df[amt_col].sum()
    avg_amount_engaged = df.loc[engaged_mask, amt_col].mean() if engaged_mask.sum() > 0 else 0
    avg_amount_new = df.loc[new_mask, amt_col].mean() if new_mask.sum() > 0 else 0

    return {
        "month": str(month),
        "buyers_total": int(buyers.sum()),
        "engaged_count": int(engaged_mask.sum()),
        "engaged_share": engaged_mask.mean(),
        "engaged_amt_share": df.loc[engaged_mask, amt_col].sum() / total_amount if total_amount > 0 else 0,
        "avg_amount_engaged": avg_amount_engaged,
        "new_count": int(new_mask.sum()),
        "new_share": new_mask.mean(),
        "new_amt_share": df.loc[new_mask, amt_col].sum() / total_amount if total_amount > 0 else 0,
        "avg_amount_new": avg_amount_new,
        "other_count": int(other_mask.sum()),
        "other_share": other_mask.mean(),
        "other_amt_share": df.loc[other_mask, amt_col].sum() / total_amount if total_amount > 0 else 0,
        "avg_amount_other": df.loc[other_mask, amt_col].mean() if other_mask.sum() > 0 else 0,
        "total_amount": total_amount
    }

def churn_metrics(df, month, months):
    # Расчет оттока с фокусом на клиентах, активных в предыдущем месяце
    tr_col = make_col("TR", month)
    amt_col = make_col("AMT", month)

    if tr_col not in df or amt_col not in df:
        return None

    # Находим предыдущий месяц
    prev_month = max([m for m in months if m < month], default=None)
    if not prev_month:
        return None

    prev_tr_col = make_col("TR", prev_month)
    prev_amt_col = make_col("AMT", prev_month)

    if prev_tr_col not in df or prev_amt_col not in df:
        return None

    # Определяем отток: активен в предыдущем месяце, но не в текущем
    was_active_prev_month = ~df[prev_tr_col].isna()
    churn_mask = was_active_prev_month & df[tr_col].isna()

    # Рассчитываем потери
    lost_revenue = df[prev_amt_col][churn_mask].sum()
    total_prev_revenue = df[prev_amt_col].sum()

    # Дополнительные метрики
    active_prev = was_active_prev_month.sum()
    churn_rate = churn_mask.sum() / active_prev if active_prev > 0 else 0
    revenue_pct = lost_revenue / total_prev_revenue if total_prev_revenue > 0 else 0

    return {
        "month": str(month),
        "prev_month": str(prev_month),
        "churn_count": int(churn_mask.sum()),
        "active_prev": int(active_prev),
        "churn_rate": float(churn_rate),
        "lost_revenue": float(lost_revenue),
        "lost_revenue_pct": float(revenue_pct),
        "avg_lost_per_client": float(lost_revenue / churn_mask.sum()) if churn_mask.sum() > 0 else 0.0
    }

# ------Основной расчёт-----

# Собираем список месяцев по TR/AMT/FLG
months = sorted({
    parse_period(c)
    for c in df.columns
    if c.startswith(("TR_", "AMT_", "FLG_"))
})

# Динамика вовлечённых и новых
dyn = pd.DataFrame([monthly_metrics(df, m, months) for m in months if monthly_metrics(df, m, months) is not None])

# Динамика оттока
churn_dyn = pd.DataFrame([churn_metrics(df, m, months) for m in months if churn_metrics(df, m, months) is not None])

# -----Анализ для февраля 2024-----

feb_2024 = pd.Period('2024-02', freq='M')
feb_data = dyn[dyn['month'] == '2024-02'].iloc[0]

print("\nАнализ февральских вовлеченных клиентов")
print(f"Всего активных клиентов: {feb_data['buyers_total']}")
print(f"Вовлеченные клиенты: {feb_data['engaged_count']} ({feb_data['engaged_share']:.1%})")
print(f"Их вклад в выручку: {feb_data['engaged_amt_share']:.1%}")
print(f"Средний чек вовлеченных: {feb_data['avg_amount_engaged']:,.0f}")
print(f"Средний чек новых клиентов: {feb_data['avg_amount_new']:,.0f}")
print(f"Средний чек остальных: {feb_data['avg_amount_other']:,.0f}")


# Сводные таблицы

def has(prefix, m): return make_col(prefix, m) in df.columns

clients_summary = {"Новые": {}, "Вовлеченные": {}, "Остальные": {}, "Итого": {}, "Отток": {}}
revenue_summary = {"Новые": {}, "Вовлеченные": {}, "Остальные": {}, "Итого": {}}

for i, m in enumerate(months):
    tr_col  = make_col("TR",  m)
    amt_col = make_col("AMT", m)
    if tr_col not in df.columns or amt_col not in df.columns:
        continue

    # Новые
    prior_tr_cols = [make_col("TR", pm) for pm in months if pm < m and make_col("TR", pm) in df]
    new_mask = (df[prior_tr_cols].fillna(0).sum(axis=1) == 0) & (df[tr_col] > 0)

    # Вовлеченные (если есть 3 месяца)
    prev3 = [months[i-k] for k in [1,2,3] if i-k >= 0]
    engaged_mask = pd.Series(False, index=df.index)
    if len(prev3) == 3 and all(has("TR", pm) for pm in prev3):
        engaged_mask = (df[[make_col("TR", pm) for pm in prev3]] > 0).all(axis=1)

    buyers_mask = (df[tr_col] > 0)
    other_mask  = buyers_mask & ~new_mask & ~engaged_mask

    # Клиенты
    clients_summary["Новые"][m]        = int(new_mask.sum())
    clients_summary["Вовлеченные"][m]  = int(engaged_mask.sum())
    clients_summary["Остальные"][m]    = int(other_mask.sum())
    clients_summary["Итого"][m]        = int(buyers_mask.sum())

    # Выручка
    revenue_summary["Новые"][m]        = float(df.loc[new_mask,     amt_col].sum())
    revenue_summary["Вовлеченные"][m]  = float(df.loc[engaged_mask, amt_col].sum())
    revenue_summary["Остальные"][m]    = float(df.loc[other_mask,   amt_col].sum())
    revenue_summary["Итого"][m]        = float(df.loc[buyers_mask,  amt_col].sum())

    # Отток % (от активных прошлого месяца)
    if i == 0:
        clients_summary["Отток"][m] = 0.0
    else:
        prev_tr = make_col("TR", months[i-1])
        prev_total = (df[prev_tr].fillna(0) > 0).sum()
        churn_mask = (df[prev_tr].fillna(0) > 0) & (df[tr_col].fillna(0) == 0)
        churn_pct = (churn_mask.sum() / prev_total * 100) if prev_total > 0 else 0.0
        clients_summary["Отток"][m] = churn_pct


clients_table = pd.DataFrame(clients_summary).T
revenue_table = pd.DataFrame(revenue_summary).T

print("\nСводная таблица по клиентам:")
print(clients_table)

print("\nСводная таблица по выручке:")
print(revenue_table)

# -----Визуализация динамики-----

plt.figure(figsize=(14, 16))

# 1. Динамика клиентов
plt.subplot(3, 1, 1)
plt.plot(dyn["month"], dyn["engaged_count"], label="Вовлеченные", marker="o")
plt.plot(dyn["month"], dyn["new_count"], label="Новые", marker="o")
plt.plot(dyn["month"], dyn["other_count"], label="Остальные", marker="o")
plt.title("Динамика количества клиентов по сегментам")
plt.ylabel("Количество клиентов")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Доли клиентов
plt.subplot(3, 1, 2)
plt.plot(dyn["month"], dyn["engaged_share"], label="Доля вовлеченных", marker="o")
plt.plot(dyn["month"], dyn["new_share"], label="Доля новых", marker="o")
plt.plot(dyn["month"], dyn["other_share"], label="Доля остальных", marker="o")
plt.title("Доли клиентских сегментов")
plt.ylabel("Доля от общего числа")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 3. Вклад в выручку
plt.subplot(3, 1, 3)
plt.plot(dyn["month"], dyn["engaged_amt_share"], label="Вовлеченные", marker="o")
plt.plot(dyn["month"], dyn["new_amt_share"], label="Новые", marker="o")
plt.plot(dyn["month"], dyn["other_amt_share"], label="Остальные", marker="o")
plt.title("Вклад сегментов в выручку")
plt.ylabel("Доля от общей выручки")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4. Анализ оттока

print("\nАнализ оттока клиентов:")
print(f"Средний уровень оттока: {churn_dyn['churn_rate'].mean():.1%}")
print(f"Максимальный отток: {churn_dyn['churn_rate'].max():.1%} в {churn_dyn.loc[churn_dyn['churn_rate'].idxmax(), 'month']}")
print(f"Средняя потеря на клиента: {churn_dyn['avg_lost_per_client'].mean():,.0f} Т")

plt.figure(figsize=(14, 6))

# 5. Динамика оттока

plt.figure(figsize=(16, 12))

# 5.1 Основные метрики оттока
plt.subplot(2, 2, 1)
plt.plot(churn_dyn["month"], churn_dyn["churn_rate"]*100,
         label="Уровень оттока (%)", marker='o', color='crimson', linewidth=2)
plt.plot(churn_dyn["month"], churn_dyn["lost_revenue_pct"]*100,
         label="Доля потерянной выручки (%)", marker='o', color='darkorange', linewidth=2)
plt.title("Динамика ключевых показателей оттока")
plt.ylabel("Процент")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 5.2 Абсолютные значения оттока
plt.subplot(2, 2, 2)
plt.bar(churn_dyn["month"], churn_dyn["churn_count"],
        color='lightcoral', label="Количество ушедших клиентов")
plt.plot(churn_dyn["month"], churn_dyn["active_prev"],
         marker='o', color='green', label="Активных в предыдущем месяце")
plt.title("Абсолютные показатели оттока")
plt.ylabel("Количество клиентов")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 5.3 Финансовые потери
plt.subplot(2, 2, 3)
plt.bar(churn_dyn["month"], churn_dyn["lost_revenue"]/1e6,
        color='salmon', label="Потерянная выручка")
plt.plot(churn_dyn["month"], churn_dyn["avg_lost_per_client"]/1e3,
         marker='o', color='purple', label="Средний убыток на клиента")
plt.title("Финансовые потери от оттока")
plt.ylabel("Выручка / Убыток")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

# 5.4 Сравнение с общей выручкой
plt.subplot(2, 2, 4)
plt.plot(revenue_table.columns.astype(str), revenue_table.loc["Итого"]/1e6,
         label="Общая выручка", marker='o', color='dodgerblue')
plt.bar(churn_dyn["month"], churn_dyn["lost_revenue"]/1e6,
        color='lightcoral', alpha=0.7, label="Потерянная выручка")
plt.title("Сравнение с общей выручкой")
plt.ylabel("Выручка")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle("Анализ оттока клиентов: динамика и финансовые последствия", y=1.02)
plt.show()

# Обьяснение формул расчета оттока
print("\nРасчет оттока")
print("Под оттоком подразумевается количество клиентов, которые совершили покупки в прошлом месяце, но не купили в текущем")
print("Уровень оттока = ушедшие / активные ранее ")
print("Потерянная выручка - какая выручка ушла с этими клиентами")
print("Доля потерь = потерянная выручка / общая выручка")

# Рекомендации по сегментам

print("\nРекомендации по сегментам")
print("1. Вовлеченные клиенты:")
print("- Увеличивать средний чек через кросс-продажи")
print("- Внедрить программу лояльности")
print("- Персонализировать предложения на основе истории покупок")

print("\n2. Новые клиенты:")
print("- Предлагать специальные условия для повторных покупок")
print("- Анализ каналов привлечения самых ценных новых клиентов")

print("\n3. Остальные клиенты:")
print("- Реактивация через email/SMS-кампании")
print("- Специальные предложения для возврата")
print("- Анализ причин низкой вовлеченности")

print("\n4. По оттоку:")
print("- Внедрить систему предсказания оттока")
print("- Разработать спецпредложения для групп риска")
print("- Улучшите сервис для проблемных сегментов")