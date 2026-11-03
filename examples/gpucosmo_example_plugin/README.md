# gpucosmo-example-plugin

A minimal, installable GPUCosmo likelihood plug-in. The entry point in `pyproject.toml`
registers `RdhGaussian` under the name `rdh_gaussian`:

```bash
pip install -e examples/gpucosmo_example_plugin
gpucosmo-likelihoods            # lists rdh_gaussian
```

Then any Cobaya yaml run with `gpucosmo-dali-hmc` / `gpucosmo-dali-ns` (or
`gpucosmo.pipeline.build_analysis_bundle`) can use it:

```yaml
likelihood:
  rdh_gaussian: {mean: 99.1, sigma: 1.0}
```

See `docs/adding_likelihoods.rst` in the GPUCosmo repository.
