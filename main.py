import streamlit as st
import pandas as pd
import os
import yaml

from sonAI_tools_utils import list_files_in_s3_folder, download_from_s3
from utils import list_models_in_bucket, model_version_from_path,  prepare_fn_fp_per_person_per_threshold, extract_test_dataset_name_from_csv_filepath
from plots import make_distress_no_distress_fig, make_distress_no_distress_per_person_fig, make_all_labels_fig, make_fn_fp_per_person_fig, make_tpr_fpr_roc_fig, color_1, color_2


#######################
#### STREAMLIT APP ####
#######################
st.set_page_config(layout="wide")
st.title("Model dashboard")

# model csv folder
model_bucket_name = "sonai-models"
csv_model_folder = "./csv_models"
os.makedirs(csv_model_folder, exist_ok=True)

# dataset csv folder
datasets_bucket_name = "datasets-files"
datasets_files_folder = "csv_datasets"
os.makedirs(datasets_files_folder, exist_ok=True)

# model csv download
model_list = list_models_in_bucket(model_bucket_name, prefix="distress_hm")
minimum_version = "1.0.7"
# models_to_keep = ["1.0.4"]
models_to_keep = []
model_list = [model_path for model_path in model_list if (model_version_from_path(model_path) >= minimum_version or model_version_from_path(model_path) in models_to_keep)]
model_path = st.sidebar.selectbox("Model", model_list, 0) # streamlit selectbox for models
model_files = [f for f in list_files_in_s3_folder(model_bucket_name, model_path) if (f.endswith(".csv") or f.endswith(".yaml"))]

for f in model_files:
    if not os.path.exists(os.path.join(csv_model_folder, f)):
        download_from_s3(model_bucket_name, f, os.path.join(csv_model_folder, f))

# train and valid dataset csv files
model_yaml = [f for f in model_files if f.endswith(".yaml")][0]
with open(os.path.join(csv_model_folder, model_yaml)) as f:
    model_config = yaml.safe_load(f)
train_dataset_name = os.path.basename(model_config["data"]["dataset"])
train_dataset_csv_file = [f for f in list_files_in_s3_folder(datasets_bucket_name) if f"{train_dataset_name}_train" in f][0]
valid_dataset_csv_file = [f for f in list_files_in_s3_folder(datasets_bucket_name) if f"{train_dataset_name}_valid" in f][0]

if not os.path.exists(os.path.join(datasets_files_folder, train_dataset_csv_file)):
    download_from_s3(datasets_bucket_name, train_dataset_csv_file, os.path.join(datasets_files_folder, train_dataset_csv_file))
if not os.path.exists(os.path.join(datasets_files_folder, valid_dataset_csv_file)):
    download_from_s3(datasets_bucket_name, valid_dataset_csv_file, os.path.join(datasets_files_folder, valid_dataset_csv_file))

# test datasets
model_csv_inference_files = [f for f in model_files if f.endswith(".csv")]
test_dataset_name_marker = "val_threshold_stud="
suffix = "_inferences.csv"
test_datasets_used_in_existing_inference = set([extract_test_dataset_name_from_csv_filepath(f, test_dataset_name_marker, suffix) for f in model_csv_inference_files if "best_model" not in f])

#TODO select test dataset if multiple test dataset inferences
test_datasets_csv_files = [f"{dataset_name}_test_metadatas.csv" for dataset_name in test_datasets_used_in_existing_inference]
for csv_file in test_datasets_csv_files:
    if not os.path.exists(os.path.join(datasets_files_folder, csv_file)):
        download_from_s3(datasets_bucket_name, csv_file, os.path.join(datasets_files_folder, csv_file))

test_dataset_csv_file = st.sidebar.selectbox("Test dataset", test_datasets_csv_files, 0) # streamlit selectbox for test dataset
test_dataset_name = test_dataset_csv_file.split("_test_metadatas")[0]
model_csv_inference_files = [f for f in model_csv_inference_files if test_dataset_name in f]

##### DATASETS INSIGHTS #####
st.subheader("Train, validation, and test datasets")
col_train, col_valid, col_test = st.columns(3)
col_train.subheader("Train")
col_valid.subheader("Validation")
col_test.subheader("Test")

# distress/no-distress plots
df_train = pd.read_csv(os.path.join(datasets_files_folder, train_dataset_csv_file))
fig_train_distress_no_distress = make_distress_no_distress_fig(df_train)
col_train.pyplot(fig_train_distress_no_distress)

df_valid = pd.read_csv(os.path.join(datasets_files_folder, valid_dataset_csv_file))
fig_valid_distress_no_distress = make_distress_no_distress_fig(df_valid)
col_valid.pyplot(fig_valid_distress_no_distress)

df_test = pd.read_csv(os.path.join(datasets_files_folder, test_dataset_csv_file))
fig_test_distress_no_distress = make_distress_no_distress_fig(df_test, fig_width=2, fig_height=2)
col_test.pyplot(fig_test_distress_no_distress)

# distress/no-distress per person plots
fig_train_distress_no_distress_per_person = make_distress_no_distress_per_person_fig(df_train, fig_width=10, fig_height=4)
col_train.pyplot(fig_train_distress_no_distress_per_person)

fig_valid_distress_no_distress_per_person = make_distress_no_distress_per_person_fig(df_valid, fig_width=10, fig_height=4)
col_valid.pyplot(fig_valid_distress_no_distress_per_person)

fig_test_distress_no_distress_per_person = make_distress_no_distress_per_person_fig(df_test, fig_width=10, fig_height=4)
col_test.pyplot(fig_test_distress_no_distress_per_person)

# train and valid all labels
fig_train_all_labels = make_all_labels_fig(df_train, bar_color=color_2, fig_width=12, fig_height=4)
col_train.pyplot(fig_train_all_labels)

fig_valid_all_labels = make_all_labels_fig(df_valid, bar_color=color_2, fig_width=10, fig_height=4)
col_valid.pyplot(fig_valid_all_labels)

fig_test_all_labels = make_all_labels_fig(df_test, bar_color=color_2, fig_width=10, fig_height=4)
col_test.pyplot(fig_test_all_labels)

##### MODEL PERFORMANCE #####
st.subheader("Model performance")
col_fn_fp, col_roc = st.columns(2)

model_csv_inference_files_basenames = [os.path.basename(f) for f in model_csv_inference_files]
model_csv = st.sidebar.selectbox("Model checkpoint", model_csv_inference_files, 0) # streamlit selectbox for model checkpoint
df_model_inference = pd.read_csv(os.path.join(csv_model_folder, model_csv))

thresholds = [0.0, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0]
threshold = st.sidebar.selectbox("Threshold", thresholds, 1) # streamlit selectbox for thresholds

# fp/fn per person in test set
df_final_fp_fn_per_person = prepare_fn_fp_per_person_per_threshold(df_model_inference, df_test, thresholds)
fig_fn_fp_per_person = make_fn_fp_per_person_fig(df_final_fp_fn_per_person, threshold, color_fn=color_1, color_fp=color_2, fig_width=7, fig_height=5)
col_fn_fp.pyplot(fig_fn_fp_per_person)


# TPR vs FPR ROC curve
threshold_labels = [str(threshold) for threshold in thresholds]
fig_tpr_fpr = make_tpr_fpr_roc_fig(df_final_fp_fn_per_person, threshold_labels=threshold_labels)
col_roc.pyplot(fig_tpr_fpr)