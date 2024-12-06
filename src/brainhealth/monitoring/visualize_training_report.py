
import argparse
import matplotlib.pyplot as plt
import numpy as np

def visualize_training_report(file_path: str):
    """
    Visualize the training report.
    
    Parameters:
    file_path (str): Path to the performance file.
    """
    with open(file_path, 'r') as file:
        dict_metrics = {}
        lines = file.readlines()
        total_steps = len(lines)
        epoches = {}

        for line in lines:
            epoch = line.split('|')[0].strip()[-1]
            if epoch not in epoches:
                epoches.update({epoch: 1})
            metrics = line.split('|')[1:-1]
            for metric in metrics:
                key = metric.split('=')[0].strip()
                value = metric.split('=')[1].strip()
                dict_metrics[key].append([np.int16(epoch),np.float32(value)]) if key in dict_metrics else dict_metrics.update({key: [[np.int16(epoch),np.float32(value)]]})
        fig, ax = plt.subplots(1, len(dict_metrics))
        count = 0
        for key, value_array in dict_metrics.items():
            ax[count].plot(range(1, total_steps + 1), [v[1] for v in value_array], label=key)
            ax[count].set_title(f'{key}')
            ax[count].set_xlabel('Steps', size=10)
            ax[count].legend()
            count += 1
        plt.show()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the training report.")
    parser.add_argument("--file_path", type=str, help="Path to the performance file.")
    args = parser.parse_args()
    
    visualize_training_report(file_path=args.file_path)