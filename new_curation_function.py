# --- imports
import re
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType, StringType
from pyspark.sql.functions import pandas_udf

# ---------- 1) Load LDAP once and broadcast ----------
def load_human_vzid_set(spark, ldap_path: str) -> set:
    ldap = spark.read.parquet(ldap_path).select("vzid").dropna().distinct()
    return set(r["vzid"].lower() for r in ldap.collect())

human_vzid_set = load_human_vzid_set(spark, "/Users/ldap_snappy.parquet")  # <— your path
b_human_vzid = spark.sparkContext.broadcast(human_vzid_set)

# ---------- 2) Your logic rewritten as a pure helper ----------
# NOTE: same rules as your screenshot, but parameterized & safely handles None.
_re_svc = re.compile(r"svc[\-_\.]", re.IGNORECASE)
# If you prefer whole-word matches for keywords, switch to regexes.
_functional_keywords = [
    "admin", "svc", "nso", "monitor", "report", "batch", "task", "job",
    "bot", "system", "api", "service", "auto", "deploy"
]

def is_likely_functional_py(user: str, human_vzid: set) -> bool:
    if not user:
        return False
    u = user.strip()
    if not u:
        return False

    u_low = u.lower()
    if u_low in human_vzid:
        return False

    # keyword heuristics
    if any(k in u_low for k in _functional_keywords):
        return True

    # pattern: svc-, svc_, svc.
    if _re_svc.search(u):
        return True

    # length heuristic
    if len(u) <= 3 or len(u) >= 10:
        return True

    # presence of special chars
    if sum(u.count(c) for c in ['_', '-', '.', ':']) >= 1:
        return True

    # camelCase / ALLCAPS-with-leading-V nuance from your code
    if u != u_low and u != u.upper() and re.search(r"[A-Z]", u):
        if u.upper().startswith("V"):  # your exception
            return False
        return True

    return False

# ---------- 3) Vectorized UDF ----------
@pandas_udf("boolean")
def is_functional_udf(users: pd.Series) -> pd.Series:
    hv = b_human_vzid.value
    return users.fillna("").map(lambda x: is_likely_functional_py(x, hv))

# ---------- 4) Apply to your dataframe inside add_fields ----------
def add_fields(df, spark, resource_path, ldap_path):
    _df = df  # ... keep your existing transforms above

    # (keep all your existing withColumn steps …)

    # Replace the old human_users_broadcast logic with:
    _df = _df.withColumn("is_functional", is_functional_udf(F.col("user")))
    _df = _df.withColumn(
        "user_type",
        F.when(F.col("is_functional"), F.lit("functional_user"))
         .otherwise(F.lit("human_user"))
    )

    return _df
