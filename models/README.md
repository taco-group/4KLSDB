# SANA 1.5 Preprocessing Embedding Feature

Here is the basic instruction of how to access the preprocessing embedding feature for SANA 1.5 training.

## Setup

First go to path `4KLSDB/models/sana/diffusion/data/datasets`, then find `embed_pro.py` file.

Use this file after you activate the SANA environment. You can build the environment either by:

1. **Using the yml file**: Use the environment file in `4KLSDB/envs/Sana_training.yml`
   ```bash
   conda env create -f 4KLSDB/envs/Sana_training.yml
   conda activate Sana
   ```

2. **Using official instruction**: Follow the official instruction in the SANA repo

## Usage

After you activate the environment, you can run the preprocessing code with command listed at the top of the `embed_pro.py` file.

```bash
cd 4KLSDB/models/sana/diffusion/data/datasets
python embed_pro.py [parameters]
```

Check the top of `embed_pro.py` for specific command usage and parameters.