# Setup
1. Download https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
2. Run `bash Miniconda3-latest-Linux-x86_64.sh`
3. Run `conda env create -f environment.yml`
4. Run `source activate merchant`
5. For Marketplace:
  * Edit `Line 13` in `config.py` and change `merchant_id` accordingly
  * Run `python MLMerchant.py --port 8090`
6. For demand learning: Run Piazza command
