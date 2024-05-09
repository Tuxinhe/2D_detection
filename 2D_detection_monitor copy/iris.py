import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import os
import math

def remove_small_nan_blocks(df, column, min_len=5):
    is_nan = df[column].isna()
    df['group'] = (is_nan != is_nan.shift()).cumsum()
    group_lengths = df.groupby('group').size()
    small_nan_groups = group_lengths[(group_lengths < min_len) & (group_lengths.index.isin(df.loc[is_nan, 'group']))]
    df['remove'] = df['group'].isin(small_nan_groups.index)
    cleaned_column = df.loc[~df['remove'], column].reset_index(drop=True)
    return cleaned_column  



# Load the data
file_path = r'D:\mediapipe_c110118152\depth_image_data\jason\002.csv'
file_name = os.path.splitext(os.path.basename(file_path))[0]

output_file_path = r'D:\mediapipe_c110118152\depth_image_data\jason'
json_file_path = output_file_path +"\\"+ file_name + "_data.csv"
df = pd.read_csv(file_path)
# df = pd.read_json(file_path)

# 直接drop空值
df_cleaned = df.dropna()
#平滑化座標
sigma = 0.5
smoothed_rg_y = gaussian_filter1d(df_cleaned['prev_right_heel_y'], sigma=sigma).astype(int)
smoothed_rg_x = gaussian_filter1d(df_cleaned['prev_right_heel_x'], sigma=sigma).astype(int)
smoothed_lf_y = gaussian_filter1d(df_cleaned['prev_left_heel_y'], sigma=sigma).astype(int)
smoothed_lf_x = gaussian_filter1d(df_cleaned['prev_left_heel_x'], sigma=sigma).astype(int)

# 将平滑后的数据添加到 DataFrame 中
df_cleaned['smoothed_right_heel_y'] = smoothed_rg_y
df_cleaned['smoothed_right_heel_x'] = smoothed_rg_x
df_cleaned['smoothed_left_heel_y'] = smoothed_lf_y
df_cleaned['smoothed_left_heel_x'] = smoothed_lf_x

#計算斜率
diff = np.diff(df_cleaned['smoothed_right_heel_y'])
diff2 = np.diff(df_cleaned['smoothed_left_heel_y'])

dist = np.ones_like(diff)
dist2 = np.ones_like(diff2)

slopes = diff / dist
slopes2 = diff2 / dist2
# 将斜率数组的长度增加到与 DataFrame 的长度相同
right_heel_slopes = np.concatenate([slopes, [np.nan]])
left_heel_slopes2 = np.concatenate([slopes2, [np.nan]])
# 将斜率数组赋值给 DataFrame 列
df_cleaned['right_heel_slopes'] = right_heel_slopes
df_cleaned['left_heel_slopes'] = left_heel_slopes2

# 計算 smoothed_data 的最大值和最小值的平均
# 初始化 landing_point 列為空值
rg_first_value = df_cleaned['smoothed_right_heel_y'].iloc[0] -2
lf_first_value = df_cleaned['smoothed_left_heel_y'].iloc[0] -2

# 條件篩選並賦值
condition = (df_cleaned['right_heel_slopes'] < 1) & (df_cleaned['right_heel_slopes'] > -1) & (df_cleaned['smoothed_right_heel_y'] > rg_first_value)
df_cleaned.loc[condition, 'right_test_point_y'] = df_cleaned.loc[condition, 'smoothed_right_heel_y']
df_cleaned.loc[condition, 'right_test_point_x'] = df_cleaned.loc[condition, 'smoothed_right_heel_x']
df_cleaned.loc[condition, 'right_test_time'] = df_cleaned.loc[condition, 'time']

df_cleaned['right_landing_point_y'] = remove_small_nan_blocks(df_cleaned, 'right_test_point_y')
df_cleaned['right_landing_point_x'] = remove_small_nan_blocks(df_cleaned, 'right_test_point_x')
df_cleaned['right_landing_time'] = remove_small_nan_blocks(df_cleaned, 'right_test_time')


condition2 = (df_cleaned['left_heel_slopes'] < 1) & (df_cleaned['left_heel_slopes'] > -1) & (df_cleaned['smoothed_left_heel_y'] > lf_first_value)
df_cleaned.loc[condition2, 'left_test_point_y'] = df_cleaned.loc[condition2, 'smoothed_left_heel_y']
df_cleaned.loc[condition2, 'left_test_point_x'] = df_cleaned.loc[condition2, 'smoothed_left_heel_x']
df_cleaned.loc[condition2, 'left_test_time'] = df_cleaned.loc[condition2, 'time']

df_cleaned['left_landing_point_y'] = remove_small_nan_blocks(df_cleaned, 'left_test_point_y')
df_cleaned['left_landing_point_x'] = remove_small_nan_blocks(df_cleaned, 'left_test_point_x')
df_cleaned['left_landing_time'] = remove_small_nan_blocks(df_cleaned, 'left_test_time')

# 标记哪些地方是 NaN

rg_is_nan_x = df_cleaned['right_landing_point_x'].isna()
lf_is_nan_x = df_cleaned['left_landing_point_x'].isna()

# 计算非空值开始的索引
right_starts = (~rg_is_nan_x & rg_is_nan_x.shift(1, fill_value=True))
left_starts = (~lf_is_nan_x & lf_is_nan_x.shift(1, fill_value=True))
# 使用提供的逻辑找出从非 NaN 到 NaN 的转变点
right_ends = (rg_is_nan_x & ~rg_is_nan_x.shift(1, fill_value=False))
left_ends = (lf_is_nan_x & ~lf_is_nan_x.shift(1, fill_value=False))

# 提取这些转变点的索引
right_start_points_index = df_cleaned[right_starts].index.tolist()
left_start_points_index = df_cleaned[left_starts].index.tolist()
right_end_points_index = df_cleaned[right_ends].index.tolist()
left_end_points_index = df_cleaned[left_ends].index.tolist()

end_points_modified = (np.array(right_end_points_index)).tolist()
end_points_modified2 = (np.array(left_end_points_index)).tolist()
print(end_points_modified)
print(right_start_points_index)
right_combined_index =sorted(end_points_modified + (right_start_points_index[1:]))
left_combined_index =sorted(end_points_modified2 + (left_start_points_index[1:]))

df_cleaned.loc[right_combined_index, 'right_n_landing_point_x'] = df_cleaned['right_landing_point_x']
df_cleaned.loc[right_combined_index, 'right_n_landing_point_y'] = df_cleaned['right_landing_point_y']
df_cleaned.loc[right_combined_index, 'right_n_landing_time'] = df_cleaned['time']

df_cleaned.loc[left_combined_index, 'left_n_landing_point_x'] = df_cleaned['left_landing_point_x']
df_cleaned.loc[left_combined_index, 'left_n_landing_point_y'] = df_cleaned['left_landing_point_y']
df_cleaned.loc[left_combined_index, 'left_n_landing_time'] = df_cleaned['time']

#计算每个 start_point 到下一个 start_point 的欧式距离
count = 0
for i in range(0,(len(right_combined_index)-1),2):  # 注意这里的步长设置为 2
    current_index = right_combined_index[i]
    next_index = right_combined_index[i+1]
    count =count +1
    # 当前和下一个 start_point 的坐标
    current_point_x = df_cleaned.at[current_index, 'right_landing_point_x']
    current_point_y = df_cleaned.at[current_index, 'right_landing_point_y']
    current_point_time = df_cleaned.at[current_index, 'time']
    next_point_x = df_cleaned.at[next_index, 'right_landing_point_x']
    next_point_y = df_cleaned.at[next_index, 'right_landing_point_y']
    next_point_time = df_cleaned.at[next_index, 'time']

    # 计算欧式距离
    if pd.notna(next_point_x) and pd.notna(next_point_y):
        right_distance = np.sqrt((next_point_x - current_point_x)**2 + (next_point_y - current_point_y)**2)
        right_step_time = next_point_time - current_point_time
        
        df_cleaned.at[next_index, 'right_step_length'] = right_distance
        df_cleaned.at[next_index, 'right_step_time'] = right_step_time
count2 = 0        
for i in range(0,(len(left_combined_index)-1),2):  # 注意这里的步长设置为 2
    current_index = left_combined_index[i]
    next_index = left_combined_index[i+1]
    count2 =count2 +1
    # 当前和下一个 start_point 的坐标
    current_point_x = df_cleaned.at[current_index, 'left_landing_point_x']
    current_point_y = df_cleaned.at[current_index, 'left_landing_point_y']
    current_point_time = df_cleaned.at[current_index, 'time']
    next_point_x = df_cleaned.at[next_index, 'left_landing_point_x']
    next_point_y = df_cleaned.at[next_index, 'left_landing_point_y']
    next_point_time = df_cleaned.at[next_index, 'time']

    # 计算欧式距离
    if pd.notna(next_point_x) and pd.notna(next_point_y):
        left_distance = np.sqrt((next_point_x - current_point_x)**2 + (next_point_y - current_point_y)**2)
        left_step_time = next_point_time - current_point_time
        
        df_cleaned.at[next_index, 'left_step_length'] = left_distance
        df_cleaned.at[next_index, 'left_step_time'] = left_step_time

right_distance_travelled = math.sqrt((df_cleaned['right_n_landing_point_x'].iloc[1]-df_cleaned['right_n_landing_point_x'].iloc[-1])**2 +(df_cleaned['right_n_landing_point_y'].iloc[1]-df_cleaned['right_n_landing_point_y'].iloc[-1])**2)
left_distance_travelled = math.sqrt((df_cleaned['left_n_landing_point_x'].iloc[1]-df_cleaned['left_n_landing_point_x'].iloc[-1])**2 +(df_cleaned['left_n_landing_point_y'].iloc[1]-df_cleaned['left_n_landing_point_y'].iloc[-1])**2)

right_anbulation_time = df_cleaned['right_n_landing_time'].iloc[-1] - df_cleaned['right_n_landing_time'].iloc[1]
left_anbulation_time = df_cleaned['left_n_landing_time'].iloc[-1] - df_cleaned['left_n_landing_time'].iloc[1]

right_velocity = right_distance_travelled / right_anbulation_time
left_velocity = left_distance_travelled / left_anbulation_time

avg_step_time = df_cleaned['right_step_time'].sum() /count 
avg_step_length =  df_cleaned['right_step_length'].sum() /count
stance_time = df_cleaned.at[right_combined_index[0], 'right_landing_time']
swing_time = avg_step_time
# 查看處理後的 DataFrame
# print(df_cleaned[['landing_point_x', 'landing_point_y', 'euclidean_distance']])
      
df_cleaned.to_csv(json_file_path, index=False)
print("结果已保存到:", output_file_path)



