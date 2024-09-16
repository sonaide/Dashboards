import pandas as pd
from sonAI_tools_utils import _extract_person_from_filename, _extract_microphone_from_filename, list_files_in_s3_folder

def extract_unique_persIDs(df):
    df_filename = df["current_filename"].apply(lambda x: _extract_person_from_filename(x))
    return sorted(list(set(df_filename.to_list())))

def extract_unique_microphones(df):
    df_filename = df["current_filename"].apply(lambda x: _extract_microphone_from_filename(x))
    return sorted(list(set(df_filename.to_list())))

def boolean_dict_to_keylist(boolean_dict):
    return [key for key in boolean_dict if boolean_dict[key]]

def count_distress_no_distress(df):
    n_distress = df[df.associated_label == "distress"].shape[0]
    n_no_distress = df[df.associated_label == "no-distress"].shape[0]

    return n_distress, n_no_distress

def count_distress_no_distress_per_person(df):
    all_persIDs = extract_unique_persIDs(df)
    distress_per_persID = []
    no_distress_per_persID = []
    for persID in all_persIDs:
        df_persID = df[df["current_filename"].apply(lambda x: _extract_person_from_filename(x)) == persID]
        n_distress = df_persID[df_persID.associated_label == "distress"].shape[0]
        n_non_distress = df_persID[df_persID.associated_label == "no-distress"].shape[0]
        distress_per_persID.append(n_distress)
        no_distress_per_persID.append(n_non_distress)

    return distress_per_persID, no_distress_per_persID

def list_models_in_bucket(bucket_name, prefix):
    model_files_list = list_files_in_s3_folder(bucket_name, prefix)
    model_list = sorted(list(set(["/".join(f.split("/")[:-1]) for f in model_files_list])))
    return model_list

def model_version_from_path(model_path):
    model_path = model_path.strip("/")
    model_version = model_path.split("/")[-2]
    model_subversion = model_path.split("/")[-1]
    model_full_name = f"{model_version}.{model_subversion}"

    return model_full_name.split("-")[1]

def add_pers_id_in_df(df):
    df['pers_id'] = df['filename'].apply(lambda x: _extract_person_from_filename(x))
    return df

def get_fp_fn_per_person(df, threshold):
    df['is_fp'] = ((df["associated_label"] == "no-distress") & (df["output_probas"] >= threshold))
    df['is_fn'] = ((df["associated_label"] == "distress") & (df["output_probas"] < threshold))
    df['is_tn'] = ((df["associated_label"] == "no-distress") & (df["output_probas"] < threshold))
    df['is_tp'] = ((df["associated_label"] == "distress") & (df["output_probas"] >= threshold))
    # This code group the dataframe by pers_id and do aggregation on 'is_fp' and 'is_fn' column to calculate the number of
    # fp and fn per person
    grouped = df.groupby('pers_id').agg(
        number_fp=('is_fp', 'sum'),
        number_fn=('is_fn', 'sum'),
        number_tn=('is_tn', 'sum'),
        number_tp=('is_tp', 'sum'),
        total_instances=('pers_id', 'count')
    ).reset_index()

    # Calculate the rate of FP and FN
    grouped['rate_fp'] = round((grouped['number_fp'] / (grouped['number_fp'] + grouped['number_tn'])), 2)
    grouped['rate_fn'] = round((grouped['number_fn'] / (grouped['number_tp'] + grouped['number_fn'])), 2)
    grouped['rate_tp'] = round((grouped['number_tp'] / (grouped['number_tp'] + grouped['number_fn'])), 2)

    grouped['threshold'] = threshold
    # Calculate the total number of fp, fn, tn and tp
    total_fp_fn_tn_tp = grouped[['number_fp', 'number_fn', 'number_tn', 'number_tp']].sum()
    total_fp_fn_tn_tp['pers_id'] = 'all'
    total_fp_fn_tn_tp['threshold'] = threshold

    # Calculate the rate of FP and FN for all pers_id
    total_fp_fn_tn_tp['rate_fp'] = round((total_fp_fn_tn_tp['number_fp'] / (total_fp_fn_tn_tp['number_fp'] + total_fp_fn_tn_tp['number_tn'])), 2)
    total_fp_fn_tn_tp['rate_fn'] = round((total_fp_fn_tn_tp['number_fn'] / (total_fp_fn_tn_tp['number_tp'] + total_fp_fn_tn_tp['number_fn'])), 2)
    total_fp_fn_tn_tp['rate_tp'] = round((total_fp_fn_tn_tp['number_tp'] / (total_fp_fn_tn_tp['number_tp'] + total_fp_fn_tn_tp['number_fn'])), 2)
    total_fp_fn_tn_tp["total_instances"] = total_fp_fn_tn_tp['number_fp'] + total_fp_fn_tn_tp['number_fn'] + total_fp_fn_tn_tp['number_tn'] + total_fp_fn_tn_tp['number_tp']

    grouped.loc[len(grouped)] = total_fp_fn_tn_tp

    # Select only the necessary columns
    return grouped[['pers_id', 'threshold', 'rate_fp', 'rate_tp', 'number_fp', 'number_fn']]

def prepare_fn_fp_per_person_per_threshold(df_model_inference, df_test, thresholds):
    df_test = df_test.rename(columns={'current_filename': 'filename'})

    df_inference_and_real_label = pd.merge(df_model_inference, df_test, on="filename", how="left")
    df_inference_and_real_label.sort_values(by=["filename"], inplace=True)

    # Add pers_id as a column for easier data manipulation
    df_inference_and_real_label = add_pers_id_in_df(df_inference_and_real_label)

    df_final_fp_fn_per_person = pd.DataFrame()
    for threshold in thresholds:
        df_fp_fn_per_person = get_fp_fn_per_person(df_inference_and_real_label, threshold)
        df_final_fp_fn_per_person = pd.concat([df_final_fp_fn_per_person, df_fp_fn_per_person])

    return df_final_fp_fn_per_person
