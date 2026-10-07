.PHONY: all data validate train-pilot train-all test sync-space run-demo clean

all: validate test

data:
	python3 scripts/fetch_data.py

validate:
	python3 scripts/validate_data.py

train-pilot:
	python3 scripts/train_bundle.py --lang ita --regime 100

train-all:
	python3 scripts/train_all_models.py

test:
	python3 -m unittest discover -s tests -v

sync-space:
	python3 scripts/sync_space.py

run-demo: sync-space
	cd hf_space && python3 app.py

clean:
	rm -rf scratch/ tmp/ build/ dist/ *.egg-info
