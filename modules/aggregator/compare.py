from .functions.compare_util import RoadAssetComparator, generate_report
import pandas as pd

# --- EXECUTION ---

def go_compare(
        set0_file='outpvmt_det_set0.csv',
        set1_file='outpvmt_det_set1.csv',
        threshold_meters=20.0,
        output_file='road_asset_comparison_report.xlsx'
):
    comparator = RoadAssetComparator(set0_file, set1_file)
    results = comparator.compare_sets(threshold_meters=threshold_meters)
    df_changes, df_unchanged, df_removed, df_added = generate_report(results)

    print("\n--- REPORT SUMMARY ---")
    print(f"Matched:   {len(results['matches_analyzed'])} (Changed: {len(df_changes)}, Unchanged: {len(df_unchanged)})")
    print(f"Removed:   {len(df_removed)}")
    print(f"Added:     {len(df_added)}")

    print("\n--- SAMPLE CHANGES ---")
    if not df_changes.empty:
        cols = ['set0_id', 'set1_id', 'change_details', 'z_set0', 'z_set1']
        print(df_changes[cols].head(5).to_string())
    else:
        print("No changes detected.")

    # --- EXPORT TO EXCEL ---
    print(f"\nSaving report to {output_file}...")
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_changes.to_excel(writer, sheet_name='Changes', index=False)
            df_unchanged.to_excel(writer, sheet_name='Unchanged', index=False)
            df_removed.to_excel(writer, sheet_name='Removed (Set 0 Only)', index=False)
            df_added.to_excel(writer, sheet_name='Added (Set 1 Only)', index=False)
        print("Done. File saved successfully.")
    except Exception as e:
        print(f"Error saving Excel file: {e}")