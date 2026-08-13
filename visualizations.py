import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')


def merge_rfm(df, rfm, key='Customer ID'):
    """Merge the transactional df with the rfm segment table on Customer ID."""
    return df.merge(rfm[[key, 'Customer Segment']], on=key, how='left')


# ============================================================
# 1. Customer Segmentation by RFM Score
# ============================================================

def plot_rfm_segment_distribution(rfm, segment_col='Customer Segment'):
    segment_counts = rfm[segment_col].value_counts()
 
    plt.figure(figsize=(8, 8))
    plt.pie(
        segment_counts,
        labels=segment_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette('viridis', n_colors=len(segment_counts))
    )
    plt.title('Customer Segmentation Distribution')
    plt.ylabel('')
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Dominant RFM Segment per City/State
# ============================================================

def plot_dominant_segment_heatmap(merged_df, region_col='State', segment_col='Customer Segment'):
    # crosstab: rows = region, columns = segment, values = counts
    pivot = pd.crosstab(merged_df[region_col], merged_df[segment_col])

    # (optional) sort rows by total count عشان الترتيب يبقى أوضح
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    plt.figure(figsize=(max(8, pivot.shape[1] * 1.2), max(6, pivot.shape[0] * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='mako', linewidths=0.5, cbar_kws={'label': 'Count'})
    plt.title(f'{segment_col} Distribution by {region_col}')
    plt.xlabel(segment_col)
    plt.ylabel(region_col)
    plt.tight_layout()
    plt.show()

    return pivot


# ============================================================
# 3. Revenue by City/State
# ============================================================

def plot_revenue_by_region(df, region_col='State', revenue_col='Sales', top_n=10):
    revenue = (
        df.groupby(region_col)[revenue_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=revenue.values, y=revenue.index, palette='rocket')
    plt.title(f'Top {top_n} {region_col}s by Revenue')
    plt.xlabel('Total Revenue')
    plt.ylabel(region_col)
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Revenue Trend Over Time
# ============================================================

def plot_revenue_trend(df, date_col='Order Date', revenue_col='Sales', freq='ME'):
    trend = (
        df.set_index(date_col)
        .resample(freq)[revenue_col]
        .sum()
    )

    plt.figure(figsize=(12, 6))
    trend.plot(marker='o')
    plt.title('Revenue Trend Over Time')
    plt.xlabel('Date')
    plt.ylabel('Total Revenue')
    plt.tight_layout()
    plt.show()

# ============================================================
# 5. Top Products/Categories by Revenue
# ============================================================

def plot_top_products_revenue_vs_profit(df, product_col='Sub-Category', 
                                          revenue_col='Sales', profit_col='Profit', top_n=10):
    grouped = df.groupby(product_col)[[revenue_col, profit_col]].sum()

    top_revenue = grouped[revenue_col].sort_values(ascending=False).head(top_n)
    top_profit = grouped[profit_col].sort_values(ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.barplot(x=top_revenue.values, y=top_revenue.index, palette='crest', ax=axes[0])
    axes[0].set_title(f'Top {top_n} {product_col}s by Revenue')
    axes[0].set_xlabel('Total Revenue')
    axes[0].set_ylabel(product_col)

    sns.barplot(x=top_profit.values, y=top_profit.index, palette='flare', ax=axes[1])
    axes[1].set_title(f'Top {top_n} {product_col}s by Profit')
    axes[1].set_xlabel('Total Profit')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()

    return grouped.sort_values(revenue_col, ascending=False)

# ============================================================
# 6. Average Order Value (AOV) by RFM Segment
# ============================================================

def plot_aov_by_segment(merged_df, segment_col='Customer Segment', revenue_col='Sales', order_col='Order ID'):
    aov = (
        merged_df.groupby(segment_col)
        .apply(lambda x: x[revenue_col].sum() / x[order_col].nunique())
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=aov.values, y=aov.index, palette='flare')
    plt.title('Average Order Value (AOV) by RFM Segment')
    plt.xlabel('AOV')
    plt.ylabel('Segment')
    plt.tight_layout()
    plt.show()



def plot_discount_vs_profit_margin(df, discount_col='Discount', margin_col='Profit Margin'):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=discount_col, y=margin_col, alpha=0.4)
    sns.regplot(data=df, x=discount_col, y=margin_col, scatter=False, color='red')
    plt.title('Discount vs Profit Margin')
    plt.xlabel('Discount')
    plt.ylabel('Profit Margin (%)')
    plt.tight_layout()
    plt.show()

    corr = df[[discount_col, margin_col]].corr().iloc[0, 1]
    print(f'Correlation: {corr:.3f}')




def plot_seasonality(df, time_col='Month', revenue_col='Sales'):
    seasonal = df.groupby(time_col)[revenue_col].sum().sort_index()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=seasonal.index, y=seasonal.values, palette='mako')
    plt.title(f'Total {revenue_col} by {time_col}')
    plt.xlabel(time_col)
    plt.ylabel(f'Total {revenue_col}')
    plt.tight_layout()
    plt.show()




