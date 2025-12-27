import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_traces(hdf5_file, eq_id, trace_type="acc_traces", save_fig=None):
    """
    Plots the traces for a given earthquake ID from the HDF5 file.

    Parameters:
    hdf5_file (str): Path to the HDF5 file.
    eq_id (str): Earthquake ID to plot traces for.
    trace_type (str): Type of traces to plot (default is "acc_traces").
    """
    with h5py.File(hdf5_file, "r") as file:
        data = file["data"]
        if eq_id not in data:
            print(f"EQ_ID {eq_id} not found in the HDF5 file.")
            return
        
        event = data[eq_id]
        if trace_type not in event:
            print(f"Trace type {trace_type} not found for EQ_ID {eq_id}.")
            return
        
        traces = event[trace_type][:]
        fig, ax = plt.subplots(figsize=(10, 6), nrows=3, ncols=1, dpi=450)
        for i, trace in enumerate(traces[0:3]):
            ax[i].plot(trace[:, 0], color='black', lw=1)
            ax[i].set_ylabel(trace_type)
            ax[i].set_xlim(0, len(trace))
            ax[i].set_xticks(np.arange(0, len(trace), step=1000), [0, 5, 10, 15, 20, 25])


        plt.xlabel('Time (s)')
        plt.show()
        if save_fig:
            fig.savefig(f"{save_fig}", dpi=450)

if __name__ == "__main__":
    hdf5_file = "../data/TSMIP_1999_2019_Vs30_integral.hdf5"
    eq_id = "24784"  
    trace_type = "acc_traces"
    save_fig = "trace_24784_acc_traces.png"
    plot_traces(hdf5_file, eq_id, trace_type=trace_type, save_fig=save_fig)