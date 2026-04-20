import pandas as pd
import numpy as np
import ast
from collections import Counter
from scipy.spatial import cKDTree

class RoadAssetComparator:
    def __init__(self, path_set0, path_set1):
        self.df0 = pd.read_csv(path_set0)
        self.df1 = pd.read_csv(path_set1)
        
    def is_nan(self, val):
        try:
            return np.isnan(val)
        except:
            return False

    def majority_vote_defects(self, defects_list):
        """
        User provided function.
        """
        if not defects_list:
            return {}, []

        vote_counter = {}
        contributions = {}
        list_contributions = []

        # Process each defect dict with its index.
        for idx, defect in enumerate(defects_list):
            if defect is None or self.is_nan(defect):
                continue
            
            # Ensure input is list-wrapped for the loop below
            current_item = defect if isinstance(defect, list) else [defect]
            
            if not current_item: continue

            for key, value in current_item[0].items():
                if value is None or self.is_nan(value):
                    continue
                if key not in vote_counter:
                    vote_counter[key] = Counter()
                    contributions[key] = {"Yes": [], "No": []}
                
                clean_val = str(value).strip()
                vote_counter[key][clean_val] += 1
                
                if clean_val in contributions[key]:
                    contributions[key][clean_val].append(idx)

        majority_defects = {}
        for key, counter in vote_counter.items():
            yes_votes = counter.get('Yes', 0)
            no_votes = counter.get('No', 0)
            final_vote = 'Yes' if yes_votes > no_votes else 'No'
            majority_defects[key] = final_vote
            list_contributions.append(contributions[key].get(final_vote, []))
        
        return majority_defects, list_contributions

    def parse_defects_column(self, df):
        parsed_list = []
        for val in df['defects']:
            try:
                d = ast.literal_eval(val)
                parsed_list.append([d]) 
            except:
                parsed_list.append(None)
        return parsed_list

    def consolidate_dataset(self, df, set_name):
        """
        Groups images by 'asset' ID if available.
        Prioritizes assetX/Y/Z, but falls back to projectedX/Y/Z if asset coords are NaN.
        Preserves lists of images, descriptions, and Z-coordinates.
        """
        df['parsed_defects'] = self.parse_defects_column(df)
        
        consolidated_data = []
        
        # Check if we can group by Asset ID
        has_asset_col = 'asset' in df.columns
        
        if has_asset_col:
            grouped = df.groupby('asset')
            iterator = grouped
        else:
            # If no asset column, we iterate over rows individually
            # Create a dummy iterator that mimics the groupby structure: (id, DataFrame-of-1-row)
            iterator = ((idx, df.iloc[[idx]]) for idx in df.index)

        for asset_id, group in iterator:
            # --- 1. Coordinate Logic (with Fallback) ---
            avg_x = np.nan
            avg_y = np.nan
            
            # Try asset coordinates first
            if 'assetX' in group.columns and 'assetY' in group.columns:
                avg_x = group['assetX'].mean()
                avg_y = group['assetY'].mean()
            
            # If asset coordinates are NaN (missing/failed reconstruction), fallback to projected
            if pd.isna(avg_x) or pd.isna(avg_y):
                if 'projectedX[m]' in group.columns and 'projectedY[m]' in group.columns:
                    avg_x = group['projectedX[m]'].mean()
                    avg_y = group['projectedY[m]'].mean()
            
            # Z Logic (Priority: AssetZ -> ProjectedZ -> NaN)
            avg_z = np.nan
            if 'assetZ' in group.columns:
                avg_z = group['assetZ'].mean()
            
            if pd.isna(avg_z):
                if 'projectedZ[m]' in group.columns:
                    avg_z = group['projectedZ[m]'].mean()
            
            asset_type = group['asset_type'].iloc[0] if 'asset_type' in group.columns else 'unknown'
            
            # --- 2. Defects Logic ---
            defects_list = group['parsed_defects'].tolist()
            majority_dict, _ = self.majority_vote_defects(defects_list)
            
            # --- 3. Metadata Lists ---
            images = group['file_name'].tolist() if 'file_name' in group.columns else []
            descriptions = group['descriptions'].tolist() if 'descriptions' in group.columns else []

            # If we didn't have an asset ID, use the filename or index as ID
            if not has_asset_col:
                final_id = images[0] if images else f"row_{asset_id}"
            else:
                final_id = asset_id

            # Only append if we have valid coordinates (otherwise we can't spatial match)
            # Add here later if I need to add timestamp or other metadata
            if not pd.isna(avg_x) and not pd.isna(avg_y):
                consolidated_data.append({
                    'set_origin': set_name,
                    'original_asset_id': final_id,
                    'asset_type': asset_type,
                    'x': avg_x,
                    'y': avg_y,
                    'z': avg_z,
                    'defects': majority_dict,
                    'image_count': len(group),
                    'image_list': images,
                    'description_list': descriptions
                })
            else:
                # Optional: Log skipped items due to no coordinates
                pass
            
        return pd.DataFrame(consolidated_data)

    def compare_sets(self, threshold_meters=20.0):
        print("Consolidating Set 0...")
        c0 = self.consolidate_dataset(self.df0, "set0")
        print(f"Set 0: Found {len(c0)} unique assets.")
        
        print("Consolidating Set 1...")
        c1 = self.consolidate_dataset(self.df1, "set1")
        print(f"Set 1: Found {len(c1)} unique assets.")

        # Spatial Matching (X, Y only)
        coords0 = c0[['x', 'y']].values
        coords1 = c1[['x', 'y']].values

        tree = cKDTree(coords1)
        distances, indices = tree.query(coords0, k=1)

        results = {
            'matches_analyzed': [],
            'only_in_set0': [],
            'only_in_set1': []
        }
        
        matched_indices_set1 = set()

        print("Comparing assets...")
        
        # 1. Iterate through Set 0
        for i in range(len(c0)):
            dist = distances[i]
            idx_in_1 = indices[i]
            
            asset0 = c0.iloc[i]
            
            if dist <= threshold_meters:
                # Match found
                asset1 = c1.iloc[idx_in_1]
                matched_indices_set1.add(idx_in_1)
                
                defects0 = asset0['defects']
                defects1 = asset1['defects']
                
                changes = {}
                has_change = False
                
                all_keys = set(defects0.keys()).union(defects1.keys())
                for k in all_keys:
                    val0 = defects0.get(k, 'N/A')
                    val1 = defects1.get(k, 'N/A')
                    if val0 != val1:
                        has_change = True
                        changes[k] = f"{val0} -> {val1}"
                
                results['matches_analyzed'].append({
                    'set0_id': asset0['original_asset_id'],
                    'set1_id': asset1['original_asset_id'],
                    'distance_diff': round(dist, 2),
                    'asset_type': asset0['asset_type'],
                    'has_changes': has_change,
                    'change_details': changes,
                    'location_set0': (round(asset0['x'], 2), round(asset0['y'], 2)),
                    'z_set0': asset0['z'],
                    'z_set1': asset1['z'],
                    'images_set0': asset0['image_list'],
                    'images_set1': asset1['image_list'],
                    'desc_set0': asset0['description_list'],
                    'desc_set1': asset1['description_list'],
                    'defects_set0': defects0,
                    'defects_set1': defects1
                })
            else:
                # No match
                results['only_in_set0'].append(asset0)

        # 2. Identify New Assets in Set 1
        for i in range(len(c1)):
            if i not in matched_indices_set1:
                results['only_in_set1'].append(c1.iloc[i])

        return results

def generate_report(results):
    # Define common columns to keep for the report
    common_cols = ['original_asset_id', 'asset_type', 'x', 'y', 'z', 'defects', 'image_list', 'description_list']

    # 1. Changes
    matches = results['matches_analyzed']
    changed_assets = [m for m in matches if m['has_changes']]
    df_changes = pd.DataFrame(changed_assets)
    
    # 2. Unchanged
    unchanged_assets = [m for m in matches if not m['has_changes']]
    df_unchanged = pd.DataFrame(unchanged_assets)
    
    # 3. Removed (Set 0 Only)
    df_removed = pd.DataFrame(results['only_in_set0'])
    if not df_removed.empty:
        df_removed = df_removed[common_cols]
    
    # 4. Added (Set 1 Only)
    df_added = pd.DataFrame(results['only_in_set1'])
    if not df_added.empty:
        df_added = df_added[common_cols]

    return df_changes, df_unchanged, df_removed, df_added