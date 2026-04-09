
from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field
from feast.types import Float64, Int64

transaction_entity = Entity(
    name="entity_id",
    description="User/account identifier",
)

tx_source = FileSource(
    path="K:/selfhealing/config/feature_repo/data/transactions.parquet",
    timestamp_field="event_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction_entity],
    ttl=timedelta(days=90),
    schema=[
        Field(name="tx_amount", dtype=Float64),
        Field(name="tx_hour", dtype=Int64),
        Field(name="tx_day_of_week", dtype=Int64),
        Field(name="merchant_category", dtype=Int64),
        Field(name="distance_from_home", dtype=Float64),
        Field(name="distance_from_last_tx", dtype=Float64),
        Field(name="ratio_to_median_price", dtype=Float64),
        Field(name="is_chip_used", dtype=Int64),
        Field(name="is_pin_used", dtype=Int64),
        Field(name="is_online", dtype=Int64),
        Field(name="tx_frequency_1h", dtype=Float64),
        Field(name="tx_amount_avg_7d", dtype=Float64),
        Field(name="tx_amount_std_7d", dtype=Float64),
    ],
    source=tx_source,
    online=True,
)
