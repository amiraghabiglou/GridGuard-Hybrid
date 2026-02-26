# src/schemas/feature_map.py
FEATURE_TRANSLATIONS = {
    "value__linear_trend__attr_\"slope\"": "Long-term consumption trend",
    "value__longest_strike_below_mean": "Consecutive days of abnormally low usage",
    "domain__zero_consumption_ratio": "Percentage of days with zero usage",
    "value__standard_deviation": "Usage consistency/volatility"
}

THEFT_PERSONAS = {
    "Meter Bypass": "Significant drop in weekday mean while maintaining minimal baseline.",
    "Partial Shunt": "Consistent percentage reduction in overall consumption volatility.",
    "Data Manipulation": "Unnatural regularity or repeated identical consumption values."
}