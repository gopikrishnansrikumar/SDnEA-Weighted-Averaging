import os
import numpy as np
import pandas as pd
import streamlit as st
from MP.STAMP import main as STAMP
from lof.lof_main import AlgorithmArgs, main as LOF
import matplotlib.pyplot as plt
from calculate_weights import calculate_weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve


def save_results_to_excel(file_path, file_name, results):
    if os.path.exists(file_path):
        # If file exists, append the results
        existing_results = pd.read_excel(file_path)
        all_results = pd.concat([existing_results, results], ignore_index=True)
        all_results.to_excel(file_path, index=False)
    else:
        # If file doesn't exist, save the results
        results.to_excel(file_path, index=False)

    # Append the test case file name as a new column
    results['Test Case'] = file_name


def execute_algorithms(config, file_path):
    try:
        # Execute LOF script and get anomaly scores
        LOF_anomaly_scores, LOF_tpr = LOF(config)

        # Execute STAMP script and get anomaly scores
        STAMP_anomaly_scores, STAMP_tpr = STAMP(file_path)

        return LOF_anomaly_scores, LOF_tpr, STAMP_anomaly_scores, STAMP_tpr
    except Exception as e:
        st.error(f"Error in anomaly detection algorithms: {e}")
        return None, None, None, None


def plot_results(values, LOF_anomaly_scores, STAMP_anomaly_scores, avg_anomaly_scores, labels, threshold, roc_auc):
    try:
        # Create subplots
        fig, axs = plt.subplots(4, 1, figsize=(10, 20))

        # Plot values
        axs[0].plot(values, color='blue')
        axs[0].set_title('Values')

        # Plot LOF anomaly scores
        axs[1].plot(LOF_anomaly_scores, color='green')
        axs[1].set_title('LOF Anomaly Scores')

        # Plot STAMP anomaly scores
        axs[2].plot(STAMP_anomaly_scores, color='red')
        axs[2].set_title('STAMP Anomaly Scores')

        # Plot average anomaly scores
        axs[3].plot(avg_anomaly_scores, color='purple', label='Average Anomaly Scores')
        axs[3].axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
        axs[3].set_title('Average Anomaly Scores')
        axs[3].legend()

        # Show plots
        st.pyplot(fig)

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(labels, avg_anomaly_scores)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error in plotting: {e}")


def calculate_metrics(labels, binary_predictions, avg_anomaly_scores):
    try:
        # Compute evaluation metrics based on threshold-based predictions and ground truth labels
        accuracy = accuracy_score(labels, binary_predictions)
        precision = precision_score(labels, binary_predictions)
        recall = recall_score(labels, binary_predictions)
        f1 = f1_score(labels, binary_predictions)
        conf_matrix = confusion_matrix(labels, binary_predictions)
        roc_auc = roc_auc_score(labels, avg_anomaly_scores)

        # Print the results
        st.write("IDK-IF Combination:")
        st.write("Accuracy:", accuracy)
        st.write("Precision:", precision)
        st.write("Recall:", recall)
        st.write("F1 Score:", f1)
        st.write("ROC AUC:", roc_auc)
        st.write("Confusion Matrix:")
        st.write(conf_matrix)

        return roc_auc, conf_matrix
    except Exception as e:
        st.error(f"Error in calculating metrics: {e}")
        return None, None


def main_app():
    st.title("Anomaly Detection Application")

    try:
        # Provide the dataset manually
        data = pd.read_csv("/Users/gopikrishnan/PycharmProjects/Codes/generated_series/generated_series/series2_sine_amplitude_long/test.csv")

        # Extract the specified column values without headers
        values = data.iloc[1:, 1].values
        values = np.array(values, dtype=float).reshape(-1, 1)
        labels = data.iloc[1:, 2].values.astype(int)

        json_file_path = st.text_input("Enter path to JSON file",
                                       "/Users/gopikrishnan/PycharmProjects/Codes/lof/manifest.json")
        config = AlgorithmArgs.from_json_file(json_file_path)

        file_path = "/path/to/your/file"  # Update this with the actual file path needed for STAMP

        LOF_anomaly_scores, LOF_tpr, STAMP_anomaly_scores, STAMP_tpr = execute_algorithms(config, file_path)

        if LOF_anomaly_scores is not None and STAMP_anomaly_scores is not None:
            if len(STAMP_anomaly_scores) < len(LOF_anomaly_scores):
                scores = np.append(STAMP_anomaly_scores,
                                   np.zeros(len(LOF_anomaly_scores) - len(STAMP_anomaly_scores)))
                STAMP_anomaly_scores = scores

            # Set threshold for anomaly detection
            threshold = st.slider("Select threshold for anomaly detection", min_value=0.0, max_value=1.0,
                                  value=0.1, step=0.01)

            w1, w2 = calculate_weights(LOF_tpr, STAMP_tpr)
            st.write("w1(LOF): ", w1)
            st.write("w2(STAMP): ", w2)

            avg_anomaly_scores = ((LOF_anomaly_scores * w1) + (STAMP_anomaly_scores * w2)) / (w1 + w2)

            # Convert average anomaly scores to binary predictions based on the threshold
            binary_predictions = (avg_anomaly_scores >= threshold).astype(int)

            # Compute evaluation metrics based on threshold-based predictions and ground truth labels
            accuracy = accuracy_score(labels, binary_predictions)
            precision = precision_score(labels, binary_predictions)
            recall = recall_score(labels, binary_predictions)
            f1 = f1_score(labels, binary_predictions)
            conf_matrix = confusion_matrix(labels, binary_predictions)
            roc_auc = roc_auc_score(labels, avg_anomaly_scores)

            # Plot results
            plot_results(values, LOF_anomaly_scores, STAMP_anomaly_scores, avg_anomaly_scores, labels, threshold,
                         roc_auc)

            # Save results to Excel
            if st.button("Save Results to Excel"):
                file_name = "test_scores.csv"  # Assuming the file name is fixed
                results_dict = {
                    "File Name": file_name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc,
                    "Confusion Matrix": str(conf_matrix)
                }
                results_df = pd.DataFrame(results_dict, index=[0])
                output_excel_path = 'LOF+STAMP.xlsx'
                save_results_to_excel(output_excel_path, file_name, results_df)
                st.write("Results saved to Excel file.")

    except Exception as e:
        st.error(f"Error processing data: {e}")


if __name__ == "__main__":
    main_app()
