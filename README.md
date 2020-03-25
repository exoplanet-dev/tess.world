To run:

```bash
git clone https://github.com/exoplanet-dev/tess.world.git tess.world
cd tess.world
./update_toi_list.sh
conda env create --prefix ./env -f environment.yml
conda activate ./env
python -m pip install -e .
python -c "import tess_world;tess_world.run_all()"
```
