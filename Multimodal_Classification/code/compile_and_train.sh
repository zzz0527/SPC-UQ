python compile_dataset.py -c cfg/default.yml
python compile_dataset.py -c cfg/noise_sample.yml
python compile_dataset.py -c cfg/noise_label.yml
python compile_dataset.py -c cfg/noise_diversity.yml

python run_baselines.py
python run_baselines.py --noise_type diversity
python run_baselines.py --noise_type label
python run_baselines.py --noise_type sample

