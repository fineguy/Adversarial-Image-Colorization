# Training and evaluating

Please use `python <script_name> -h` to see the exact parameters for each script.

## Usage

**Training:**

`python main.py 64 64 train --use_wass`

**Testing:**

`python main.py 64 64 apply --weights <weights_path>`


### Expected outputs:

- Weights are saved in  /output/models
- Figures are saved in  /output/figures
- Save model weights every few epochs
