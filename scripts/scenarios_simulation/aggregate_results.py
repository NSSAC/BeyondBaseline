#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Aggregate metrics and compute standardized Z-scores.")
    parser.add_argument("--input-dir", default="replicate_results", 
                        help="Path to the parent directory containing all the replicate_* folders.")
    parser.add_argument("--outdir", default=None, 
                        help="Where to save the aggregated CSVs. Defaults to the same folder as --input-dir.")
    args = parser.parse_args()

    input_base = Path(args.input_dir)
    outdir_base = Path(args.outdir) if args.outdir else input_base
    outdir_base.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f" Aggregating AUC rankings from: {input_base}")
    print(f"{'='*60}")
    
    # =========================================================================
    # 1. AGGREGATE AUC RANKINGS & CALCULATE Z-SCORES
    # =========================================================================
    auc_files = list(input_base.glob("replicate_*/AUC_rankings.csv"))
    if auc_files:
        df_list = []
        for f in auc_files:
            df = pd.read_csv(f)
            df["replicate"] = f.parent.name
            df_list.append(df)
            
        all_auc = pd.concat(df_list, ignore_index=True)
        
        # Calculate Medians
        agg_auc = all_auc.groupby(["eval_type", "algorithm", "scenario_id", "scenario_label"]).agg(
            median_auc=("auc", "median"),
            mean_auc=("auc", "mean"),
            successful_runs=("auc", "count")
        ).reset_index()
        
        # --- Z-SCORE & RANK CALCULATION ---
        def process_auc_metrics(group):
            eval_type = group['eval_type'].iloc[0]
            
            # Identify if this metric is "higher is better" (Coverage metrics)
            higher_is_better_prefixes = ('F_', 'I_', 'J_', 'K_', 'L_', 'M_', 'N_')
            asc = not eval_type.startswith(higher_is_better_prefixes) # True if Lower is Better (KL, Error)
            
            # Rank: 1 is the best. If asc=True (lower is better), rank ascending.
            group['rank_of_median_auc'] = group['median_auc'].rank(method="dense", ascending=asc)
            
            mean_val = group['median_auc'].mean()
            std_val = group['median_auc'].std()
            
            if pd.isna(std_val) or std_val == 0:
                group['z_score'] = 0.0
            else:
                z = (group['median_auc'] - mean_val) / std_val
                
                # CRITICAL: Invert Z-score for "lower is better" metrics 
                # so that POSITIVE always means BETTER performance.
                if asc:
                    z = -z
                group['z_score'] = z
                
            return group

        # Apply the logic row-wise (grouped by metric)
        agg_auc = agg_auc.groupby("eval_type", group_keys=False).apply(process_auc_metrics)
        agg_auc = agg_auc.sort_values(["eval_type", "rank_of_median_auc", "algorithm", "scenario_id"])
        
        auc_out = outdir_base / "Aggregated_Median_AUC_Rankings.csv"
        agg_auc.to_csv(auc_out, index=False)
        print(f"Saved: {auc_out}")
    else:
        print("No AUC_rankings.csv files found.")


    # =========================================================================
    # 2. AGGREGATE MUGRATION METRICS & CALCULATE Z-SCORES
    # =========================================================================
    print(f"\n{'='*60}")
    print(f" Aggregating Mugration Metrics from: {input_base}")
    print(f"{'='*60}")

    mug_files = list(input_base.glob("replicate_*/Mugration_Metrics.csv"))
    if mug_files:
        mug_list = []
        for f in mug_files:
            df = pd.read_csv(f)
            df["replicate"] = f.parent.name
            mug_list.append(df)
            
        all_mug = pd.concat(mug_list, ignore_index=True)
        
        # Calculate Medians
        agg_mug = all_mug.groupby(["algorithm", "scenario_id", "scenario_label"]).agg(
            median_pearson=("pearson_r", "median"),
            median_cosine=("cosine_similarity", "median"),
            median_f1=("topological_f1", "median"),
            median_masked_mae=("masked_mae", "median"),
            successful_runs=("pearson_r", "count")
        ).reset_index()
        
        # --- Z-SCORE CALCULATION ---
        # Map out which metrics are "lower is better"
        # Tuple: (Column Name, is_lower_better)
        mug_metrics = [
            ('median_pearson', False),
            ('median_cosine', False),
            ('median_f1', False),
            ('median_masked_mae', True)  # MAE is an error rate, so lower is better
        ]
        
        for col, lower_is_better in mug_metrics:
            mean_val = agg_mug[col].mean()
            std_val = agg_mug[col].std()
            z_col = col.replace('median_', 'z_score_')
            
            if pd.isna(std_val) or std_val == 0:
                agg_mug[z_col] = 0.0
            else:
                z = (agg_mug[col] - mean_val) / std_val
                
                # Invert so POSITIVE always means BETTER
                if lower_is_better:
                    z = -z
                agg_mug[z_col] = z
        
        # Sort by F1-Score (Highest is best indicator of detecting true transmission bridges)
        agg_mug = agg_mug.sort_values(["scenario_id", "median_f1"], ascending=[True, False])
        
        mug_out = outdir_base / "Aggregated_Mugration_Metrics.csv"
        agg_mug.to_csv(mug_out, index=False)
        print(f"Saved: {mug_out}")
        
        print("\n=== Top Strategy per Scenario (by Median F1-Score) ===")
        top_mug = agg_mug.loc[agg_mug.groupby('scenario_id')['median_f1'].idxmax()]
        for _, row in top_mug.iterrows():
            print(f"  Scenario {row['scenario_id']} ({row['scenario_label']}):")
            print(f"    -> {row['algorithm']} - F1: {row['median_f1']:.4f} (Cosine: {row['median_cosine']:.4f}, Z-F1: {row['z_score_f1']:+.2f})")
    else:
        print("No Mugration_Metrics.csv files found. Was --abm_mugration used?")

if __name__ == "__main__":
    main()