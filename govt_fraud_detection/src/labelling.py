def apply_indian_red_flags(df):
    """
    Labels transactions based on GFR (General Financial Rules) 
    and CVC (Central Vigilance Commission) guidelines.
    """
    # Flag 1: Split Purchasing (Transactions just below ₹5 Lakhs threshold)
    df['is_split_purchase'] = ((df['amount'] > 450000) & (df['amount'] < 500000)).astype(int)
    
    # Flag 2: Single Bidder in Competitive Tender
    # In many Indian states, 3 bids are required. Single bids are anomalies.
    df['is_single_bid'] = (df['bidder_count'] == 1).astype(int)
    
    # Flag 3: Vendor-Department Familiarity (High repeat wins in short time)
    df['repeat_win_streak'] = df.groupby(['dept_id', 'vendor_id'])['id'].transform('count')
    
    # Combine for a heuristic 'fraud_score'
    df['heuristic_label'] = (df['is_split_purchase'] | df['is_single_bid']).astype(int)
    return df