# Causal Discovery in Time Series 📈

This is the repository for the paper *Time-Series Causal Discovery via Differentiable Permutations*, where we propose a new time series causal discovery method. In our experiments, we use 3 benchmarks with 11 datasets and compare our method against 12 baseline algorithms.  
Our method is implemented under the `algorithms/sinkhorn` folder.

---

## 🚀 Installation

We recommend using a virtual environment to manage dependencies. Our experiments were conducted using Python 3.10.

1. **Create and Activate a Conda Environment:**

```bash
conda create -n causal-env python=3.10.18
conda activate causal-env
```

2. **Install Required Packages:**

```bash
pip install -r requirements.txt
```

Though we provide `requirements.txt` with library versions, there could still be version and dependency errors since some libraries used in some methods are too old. You may need to fix these according to your environment settings.

---

## 📂 Project Structure

The repository is organized to separate algorithms, data, and results, with a main script to orchestrate experiments:

```
├── algorithms/
│   ├── sinkhorn/
│   │   ├── sinkhorn.py
│   │   └── csvif.py
│   ├── TCDF/
│   │   ├── TCDF/
│   │   └── csvif.py
│   └── ... (other algorithm folders)
├── data/
│   └── Antivirus_Activity/
│       ├── preprocessed_1.csv
│       ├── preprocessed_2.csv
│       └── summary_matrix.npy
│       └── ... (other dataset folders)
├── process_data/
├── result/
│   └── ... (output files will be generated here)
├── get_time_result.py
├── run.py
├── requirements.txt
└── README.md
```

- **algorithms/**: Contains implementations for each causal discovery method, including a `csvif.py` interface for command-line use.  
- **data/**: Stores experimental datasets. Each dataset has its own folder with the time series (.csv) and ground truth matrices (.npy).  
- **process_data/**: Provides helper files to preprocess the original datasets.  
- **result/**: Automatically created to store all outputs, such as adjacency lists, trained models, and evaluation results.  
- **run.py**: The main script to automate running all experiments and generating the final evaluation table.  

> **Note:** Almost all CSV data used for experiments are included in `data/`, except for the Bavaria and East Germany datasets which are too large. You can download them from: [CausalRivers](https://github.com/CausalRivers/causalrivers)

---

## 📊 Data Format

To use your own datasets, please ensure they follow the format below. The ground truth matrices use the convention where `matrix[i, j] = 1` signifies a causal link from variable **j** to variable **i** ($j \rightarrow i$).

### Dataset (`<data_name>.csv`)

A `.csv` file containing all time series data. Example:

```csv
ram_global_prct,cpu_global_prct,disk_io_read_mega_byte,disk_io_write_mega_byte,consumers_causality_1,messages_causality_1,publish_rate_causality_1
76.0,2.6,326608.52,83692.31,0,0,0.0
76.5,5.0,326608.52,83692.32,41,0,0.0
76.4,2.9,326608.52,83692.32,41,0,1.2
76.6,4.6,326608.52,83692.32,6,0,1.2
```

### Ground Truth (`summary_matrix.npy`)

A **binary** NumPy array (`.npy`) representing the complete causal graph:

- **1**: A causal link exists ($j \rightarrow i$).  
- **0**: No causal link exists.  

Example:

```python
[[0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 0]]
```

---

## ⚡ Usage

You can run the entire evaluation pipeline or execute individual algorithms from the command line.

### Run the Full Pipeline

The `run.py` script automates the entire process. It runs all configured algorithms on the specified datasets and saves the results to the `result/` directory.

```bash
python run.py
```

The `final_results_lag<lag_setting>.csv` file under `result/` contains a table of F1 score, recall, and precision metrics for all algorithms on all datasets.

> **Note:** We uncomment our method and the MOM1 dataset with lag 3 for users to test if they can run it in their environment, which will generate a `test.csv` file reporting the result. To customize which datasets or methods to run, simply edit the `paths`, `methods_to_run`, and `lags_to_test` lists inside `run.py`.

> **Note:** Some baseline methods may produce different results across runs (e.g., TiMINO, which involves heuristic or stochastic search), and therefore their outputs may not exactly match the numbers reported in the paper.
For our method, we use fixed random seeds to improve reproducibility. Under the same software and hardware environment, our implementation yields consistent results with those reported in the paper. However, due to inherent numerical differences across hardware platforms and deep learning libraries, slight variations may still occur in different computing environments.

### Run a Specific Algorithm

Each algorithm can be run individually via its command-line interface. This is useful for testing or fine-tuning a single method.

**General Command Structure:**

```bash
python algorithms/<algorithm_name>/csvif.py --input <path_to_data.csv> --output <path_to_output.txt> --time <path_to_running_time.txt> [OPTIONS]
```

**Example: Running Our Proposed Method**

```bash
python algorithms/sinkhorn/csvif.py
  --input data/Middleware_oriented_message_Activity/monitoring_metrics_1.csv
  --output result/Middleware_oriented_message_Activity/MoM_1/sinkhorn_adj_lag3.txt
  --save_path result/Middleware_oriented_message_Activity/MoM_1/sinkhorn_model_lag3.pth
  --time result/Middleware_oriented_message_Activity/MoM_1/sinkhorn_time_lag3.txt
```

To see all available options for a specific algorithm, use the `--help` flag:

```bash
python algorithms/varlingam/csvif.py --help
```
