import polars as pl
import pandas as pd
import cudf
from cuml.model_selection import train_test_split

def load_data(transactions_path: str, fx_rates_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Загрузка данных из Parquet файлов."""
    transactions = pl.read_parquet(transactions_path, low_memory=True)
    fx_rates = pl.read_parquet(fx_rates_path, low_memory=True)
    return transactions, fx_rates

def preprocess_transactions(transactions: pl.DataFrame) -> pl.DataFrame:
    """Предобработка транзакций: приведение типов и извлечение фич."""
    # Приведение типов столбцов
    transactions = transactions.with_columns([
        pl.col("timestamp").cast(pl.Datetime("us")),
        pl.col("is_fraud").cast(pl.Boolean),
        pl.col("is_high_risk_vendor").cast(pl.Boolean),
        pl.col("is_card_present").cast(pl.Boolean),
        pl.col("is_outside_home_country").cast(pl.Boolean),
        pl.col("is_weekend").cast(pl.Boolean),
        pl.col("vendor_category").cast(pl.Categorical),
        pl.col("vendor_type").cast(pl.Categorical),
        pl.col("currency").cast(pl.Categorical),
        pl.col("country").cast(pl.Categorical),
        pl.col("city").cast(pl.Categorical),
        pl.col("device").cast(pl.Categorical),
        pl.col("card_type").cast(pl.Categorical),
    ])

    # Извлечение полей из структуры last_hour_activity
    transactions = transactions.with_columns([
        pl.col("last_hour_activity").struct.field("num_transactions").alias("lh_num_trans"),
        pl.col("last_hour_activity").struct.field("unique_countries").alias("lh_unique_countries"),
        pl.col("last_hour_activity").struct.field("unique_merchants").alias("lh_unique_merchants"),
        pl.col("last_hour_activity").struct.field("total_amount").alias("lh_total_amount"),
        pl.col("last_hour_activity").struct.field("max_single_amount").alias("lh_max_single_amount"),
    ])

    # Создание колонки date (без времени)
    return transactions.with_columns(
        pl.col("timestamp").dt.truncate("1d").cast(pl.Date).alias("date")
    )

def preprocess_fx_rates(fx_rates: pl.DataFrame) -> pl.DataFrame:
    """Предобработка FX rates: преобразование в длинный формат."""
    rates = fx_rates.with_columns(pl.col("date").cast(pl.Date))
    return rates.melt(
        id_vars="date",
        variable_name="currency",
        value_name="rate_to_usd"
    ).with_columns(pl.col("currency").cast(pl.Categorical))

def join_fx_rates(transactions: pl.DataFrame, fx_rates: pl.DataFrame) -> pl.DataFrame:
    """Объединение транзакций с курсами валют."""
    # Сортировка для asof join
    transactions_sorted = transactions.sort(["currency", "date"])
    rates_sorted = fx_rates.sort(["currency", "date"])

    # Выполнение asof join
    return transactions_sorted.join_asof(
        rates_sorted,
        left_on="date",
        right_on="date",
        by="currency",
        strategy="backward",
    )

def create_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Создание новых признаков."""
    # Конвертация в USD с защитой от нулевых значений
    transactions = transactions.with_columns([
        pl.when(pl.col("rate_to_usd").is_null() | (pl.col("rate_to_usd") == 0))
          .then(None)
          .otherwise(pl.col("rate_to_usd"))
          .alias("rate_to_usd_safe"),
        (pl.col("amount") / pl.col("rate_to_usd")).alias("amount_usd")
    ])

    # Извлечение временных признаков
    transactions = transactions.with_columns([
        pl.col("timestamp").dt.hour().alias("hour"),
        pl.col("timestamp").dt.weekday().alias("day_of_week"),
        pl.col("timestamp").dt.date().alias("date_only")
    ])

    # Создание сложных признаков
    transactions = transactions.with_columns([
        # Новое устройство
        (pl.col("device_fingerprint") !=
         pl.first("device_fingerprint").over("customer_id")).alias("is_new_device"),

        # Скачок активности
        (pl.col("lh_num_trans") > 2 * pl.mean("lh_num_trans").over("customer_id"))
        .alias("activity_spike"),

        # Географические риски
        pl.col("country").is_in(["Mexico", "Russia", "Brazil", "Nigeria"]).alias("is_high_risk_country"),
        (pl.col("city") == "Unknown City").alias("is_unknown_city"),

        # Признаки суммы
        (pl.col("amount_usd") < 5).alias("is_small_amount"),
        (pl.col("amount_usd") > 5000).alias("is_large_amount"),

        # Временные признаки
        pl.col("hour").is_between(1, 4).alias("is_high_risk_hour"),

        # Логарифмированная сумма
        (pl.col("amount_usd") + 1).log().alias("log_amount")
    ])
    return transactions.with_columns([
        # Комбинированные признаки
        (pl.col("is_outside_home_country") & pl.col("is_new_device")).alias("abroad_new_device"),
    ])

def prepare_final_data(transactions: pl.DataFrame) -> pl.DataFrame:
    """Финализация данных для моделирования."""
    # Фильтрация данных (узнали из EDA, что тут дело не чисто)
    transactions = transactions.filter(pl.col("channel") != "pos")

    # Удаление ненужных колонок
    drop_columns = [
        "transaction_id", "customer_id", "card_number", "timestamp",
        "vendor", "currency", "device_fingerprint", "ip_address",
        "last_hour_activity", "date", "rate_to_usd", "rate_to_usd_safe"
    ]
    transactions = transactions.drop(drop_columns)

    # Преобразование булевых колонок в числовые
    bool_cols = ["is_card_present", "is_outside_home_country",
                 "is_high_risk_vendor", "is_weekend", "is_fraud"]
    for col in bool_cols:
        transactions = transactions.with_columns(pl.col(col).cast(pl.Int8))

    # Обработка категориальных признаков
    high_cardinality = ["country", "city"]
    for col in high_cardinality:
        transactions = transactions.with_columns(
            (pl.col(col) == "Unknown").alias(f"is_unknown_{col}")
        )
        transactions = transactions.drop(col)

    # One-Hot Encoding
    categorical_cols = ["vendor_category", "vendor_type", "card_type",
                        "device", "channel", "city_size"]
    return transactions.to_dummies(columns=categorical_cols, separator="_")

def balance_data(df: pl.DataFrame, target: str = "is_fraud") -> pd.DataFrame:
    """Балансировка данных через undersampling."""
    pandas_df = df.to_pandas()
    fraud = pandas_df[pandas_df[target] == 1]
    non_fraud = pandas_df[pandas_df[target] == 0].sample(n=len(fraud))
    return pd.concat([non_fraud, fraud])

def prepare_for_training(gdf: cudf.DataFrame, target: str = "is_fraud") -> tuple:
    """Подготовка данных для обучения в cuML."""
    X = gdf.drop(target, axis=1).astype("float32") # type: ignore
    y = gdf[target].astype("int32") # type: ignore
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )