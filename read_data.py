import pandas as pd
import os

path_to_data_set = 'data\\'


def read_preprocessed_af_data():
    """
    :return: A Pandas DataFrame with the preprocessed AF data
    """
    with open(path_to_data_set + 'Preprocessed_AFData.csv', encoding='UTF-8') as f:
        return pd.read_csv(f)


def process_raw_af_data():
    """
    Read the raw AF data. Divide it in samples of 30 seconds. Remove invalid samples
    :return: A Pandas DataFrame containing all samples with corresponding labels
    """
    # Define how to parse dates in both the data and control files
    X_dateparse = lambda dates: pd.datetime.strptime(dates, '%H:%M:%S')
    y_dateparse = lambda dates: pd.datetime.strptime(dates, '%H:%M:%S:%f')

    # Initialize a dict to store all samples
    samples = {'samples': [], 'labels': []}

    # Iterate through all Data, Control file pairs
    for X_file, y_file in zip(os.listdir(path_to_data_set + 'AF Data\\ECG_data'),
                              os.listdir(path_to_data_set + 'AF Data\\Class')):
        print('Processing {} & {}'.format(X_file, y_file))
        # Initialize a dict to store the Data file's data
        xs = {'timestamp': [], 'RR interval': [], 'Note': []}
        # Parse all data lines
        for line in open(path_to_data_set + 'AF Data\\ECG_data\\' + X_file).readlines():
            entries = line.split(' ')
            xs['timestamp'].append(X_dateparse(entries[0]))
            xs['RR interval'].append(entries[1])
            if len(entries) > 3:
                xs['Note'].append(entries[3])
            else:
                xs['Note'].append('')
        # Convert the Data file's data to a DataFrame
        X_df = pd.DataFrame.from_dict(xs).set_index('timestamp')
        # Read the Control labels to a DataFrame
        y_df = pd.read_table(path_to_data_set + 'AF Data\\Class\\' + y_file,
                             index_col=0,
                             sep='\s+',
                             names=['timestamp', 'label'],
                             parse_dates=[0],
                             date_parser=y_dateparse)

        # For each label in Control, get the corresponding time interval in Data
        for i, row in y_df.iterrows():
            sample = X_df.loc[(i <= X_df.index) & (i + pd.Timedelta(seconds=35) > X_df.index)].as_matrix()
            # Sample filter:
            # Samples labeled with -1 are invalid
            if row['label'] == -1:
                continue
            # Samples that contain pauses are invalid
            if 'Pause' in sample[:, 0]:
                continue
            # Samples that contain unphysiological high RR intervals are invalid (not sure about range)
            sample = [int(s) for s in sample[:, 1]]
            if any([200 > s or s > 1700 for s in sample]):
                continue
            # If there was not enough data in the time frame the sample is invalid
            if len(sample) < 30:
                continue
            # Cut sample to equal size
            sample = sample[:30]
            # Add samples to total
            samples['samples'].append(sample)
            samples['labels'].append(row['label'])
    # Return samples as DataFrame
    return pd.DataFrame.from_dict(samples)


def write_af_data_to(path):
    """
    Write the processed raw AF data to the specified file
    :param path: File path to write to
    """
    process_raw_af_data().to_csv(path)


def read_af_data():
    """
    :return: A Pandas DataFrame with raw AF samples with labels
    """
    return pd.read_csv('data\\AF Data\\raw_data.csv', index_col=0)


if __name__ == '__main__':
    # data = read_preprocessed_af_data()
    # print(data.head())

    write_af_data_to('data\\AF Data\\raw_data.csv')

    # df = read_af_data()
    # print(df.head())
