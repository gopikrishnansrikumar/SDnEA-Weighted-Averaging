import pandas as pd


def convert_labels(label):
    if label == 0:
        return 'nominal'
    elif label == 1:
        return 'anomaly'
    else:
        return 'unknown'


def convert_csv(input_filename, output_filename):
    # Read the original CSV file
    df = pd.read_csv(input_filename)

    # Rename columns and convert label values
    df = df.rename(columns={'value-0': 'x', 'is_anomaly': 'label'})
    df['label'] = df['label'].apply(convert_labels)

    # Save the DataFrame to a new CSV file
    df.to_csv(output_filename, columns=['label'], index=False)


# Example usage:
input_filename = '/Users/gopikrishnan/Documents/generated_series/generated_series/series2_sine_amplitude_long/test.csv'
output_filename = '/Users/gopikrishnan/Documents/generated_series/generated_series/series2_sine_amplitude_long/test_scores.csv'
convert_csv(input_filename, output_filename)
