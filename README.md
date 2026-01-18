# This is the repository for the final project of reinforcement learning course
## Setup
The codes are included in directory `code/`. To run the code, it requires for the physical engine MuJoCo 2.1.0 (https://github.com/google-deepmind/mujoco/releases?page=5).

Then install the dependencies
``` bash
pip install -r requirements.txt
```
## Run the code
Follow the instruction to run the code:
`python main.py --env ENV --device DEVICE --seed SEED --type TYPE [--eps EPS] [--beta BETA] [--dtarg DTARG] --exp
               EXP --name NAME`
1. `--env` represents for the benchmark. Use `all` to train the model on all the benchmarks.
2. Set `--seed` for random seeds.
3. `--type` means the variants of the algorithm. `c` for clipping; `a` for adaptive penalty; `f` for fixed penalty; `r` for rollback; `j` for JS
4. `--beta` for beta; `--dtarg` for $d_{targ}$; `--eps` for epsilon
5. Let `--exp` to be `1` forever.

An example
``` bash
python main.py -e all -d cuda -s 42 -t r --eps 0.2 -x 1 -n adapt_r_0.2
```
