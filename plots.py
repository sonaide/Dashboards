import matplotlib.pyplot as plt
import numpy as np

from utils import count_distress_no_distress, extract_unique_persIDs, count_distress_no_distress_per_person

# fig params
plt.rcParams.update({
    "figure.facecolor":  (1.0, 0.0, 0.0, 0.0), # transparent background
    "figure.figsize": (7, 5)
})
color_1 = "#7cbfc3"
color_2 = "#312581"

def make_distress_no_distress_fig(df_dataset, title="", color="white", fig_width=3, fig_height=3):
    fig, ax = plt.subplots(1)
    n_audio = df_dataset.shape[0]
    n_distress, n_no_distress= count_distress_no_distress(df_dataset)
    distress_labels = ["distress", "no-distress"]
    ax.pie([n_distress, n_no_distress], labels=distress_labels, autopct=lambda x: f"{int(x/100*n_audio):d}", textprops={"color":color}, colors=[color_2, color_1])
    ax.axis('equal') 

    ax.set_title(title, color=color)
    fig.set_figheight(fig_height)
    fig.set_figwidth(fig_width)

    return fig

def make_distress_no_distress_per_person_fig(df_dataset, title="", color="white", fig_width=5, fig_height=2):
    fig, ax = plt.subplots(1)

    all_persIDs = extract_unique_persIDs(df_dataset)
    distress_per_persID, no_distress_per_persID = count_distress_no_distress_per_person(df_dataset)

    bar_distress_per_person = ax.bar(all_persIDs, distress_per_persID, color=color_1)
    bar_no_distress_per_person = ax.bar(all_persIDs, no_distress_per_persID, bottom=distress_per_persID, color=color_2)
    ax.bar_label(bar_distress_per_person, label_type='center', color=color)
    ax.bar_label(bar_no_distress_per_person, label_type='center', color=color)
    ax.set_ylabel("Number of audio files", color=color)
    ax.set_xlabel("PersID", color=color)
    ax.tick_params(color=color, labelcolor=color)
    ax.set_title(title, color=color)
    ax.legend(["Distress", "No-distress"])

    plt.xticks(rotation=90)
    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)

    return fig

def make_all_labels_fig(df_dataset, title="", text_color="white", bar_color="blue", fig_width=10, fig_height=4):
    fig, ax = plt.subplots(1)

    labels_data = {}
    for label_list in df_dataset.true_label:
        if type(label_list) == float:
            continue
        for label in label_list.split(';'):
            if label in labels_data:
                labels_data[label] += 1
            else:
                labels_data[label] = 1
    labels_labels = list(labels_data.keys())
    labels_count = list(labels_data.values())

    bar_all_labels = ax.bar(labels_labels, labels_count, color=bar_color)  
    ax.bar_label(bar_all_labels, label_type='center', color=text_color)
    ax.set_ylabel("Number of audio files", color=text_color)
    ax.tick_params(color=text_color, labelcolor=text_color)
    ax.set_title(title, color=text_color)
    ax.tick_params(axis='x', labelrotation=90)
    ax.set_title(title, color=text_color)

    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)

    return fig

def make_fn_fp_per_person_fig(df_final_fp_fn_per_person, threshold, color_fn="lightblue", color_fp="darkblue", text_color='white', bar_width=0.2, fig_width=10, fig_height=4):
    fig, ax = plt.subplots()
    df_fp_fn_per_person = df_final_fp_fn_per_person[df_final_fp_fn_per_person["threshold"] == threshold]
    xticks_positions = np.arange(df_fp_fn_per_person['pers_id'].shape[0])

    ax.bar(xticks_positions + bar_width, df_fp_fn_per_person['number_fp'], width=0.4, label='False positives', color=color_fp)
    ax.bar(xticks_positions - bar_width, df_fp_fn_per_person['number_fn'], width=0.4, label='False negatives', color=color_fn)
    ax.set_xlabel('PersID', color=text_color)
    ax.set_xticks(xticks_positions + bar_width//2, df_fp_fn_per_person["pers_id"], color=text_color)
    ax.tick_params(color=text_color, labelcolor=text_color)
    ax.set_ylabel('Number of false positives and negatives', color=text_color)
    ax.set_title(f'Threshold = {threshold}', color=text_color)
    ax.legend()

    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)

    return fig

def make_tpr_fpr_roc_fig(df_final_fp_fn_per_person, color="white", fig_width=5, fig_height=5):
    all_persIDs = df_final_fp_fn_per_person['pers_id'].unique()

    fig, ax = plt.subplots()
    for person in all_persIDs:
        if person == "all":
            linestyle = "-"
        else:
            linestyle = "--"
        person_df_fpr_tpr = df_final_fp_fn_per_person[df_final_fp_fn_per_person['pers_id'] == person]
        ax.plot(person_df_fpr_tpr['rate_fp'], person_df_fpr_tpr['rate_tp'], marker='.', linestyle=linestyle, label=person)

    ax.set_title(f'ROC Curve (FPR vs TPR) for person {person}')
    ax.set_xlabel('False positive rate (FPR)', color=color)
    ax.set_ylabel('True positive rate (TPR)', color=color)
    ax.tick_params(color=color, labelcolor=color)
    ax.legend(all_persIDs)
    ax.grid(True)

    fig.set_figwidth(fig_width)
    fig.set_figheight(fig_height)

    return fig
