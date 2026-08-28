PYTHON := python3
COREPY := env PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX=.pycache $(PYTHON)

.PHONY: test verify-core verify-scalable verify

test:
	$(COREPY) -m unittest discover -s tests -v

verify-core: test
	$(COREPY) scripts/verify_exact_sign_flip.py
	$(COREPY) scripts/verify_two_outer_optima.py
	$(COREPY) scripts/verify_near_optimal_breakdown.py
	$(COREPY) scripts/verify_synth_demo_k2.py
	$(COREPY) scripts/verify_predictor_floor_synthetic.py
	$(COREPY) scripts/verify_manifest.py

verify-scalable:
	PYTHONWARNINGS=ignore uv run --frozen --extra scalable python -m unittest tests.test_scalable_qp tests.test_prior_comparison -v
	PYTHONWARNINGS=ignore uv run --frozen --extra scalable python scripts/verify_basque_k2_scalable.py
	PYTHONWARNINGS=ignore uv run --frozen --extra scalable python scripts/verify_basque_full.py
	PYTHONWARNINGS=ignore uv run --frozen --extra scalable python scripts/verify_prop99_full.py
	PYTHONWARNINGS=ignore uv run --frozen --extra scalable python scripts/verify_prior_comparison.py

verify: verify-core verify-scalable
