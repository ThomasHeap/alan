from setuptools import setup, find_packages

setup(
    name = "alan",
    version = "0.0.1",
    keywords = ("test", "xxx"),
    license = "MIT Licence",
    packages=['alan'],
    package_dir={'':'src'},
    python_requires='>=3.9.0',
    install_requires=[
        # <2.4 pin: alan's data/covariate ingestion (BoundPlate.expand_named,
        # named2dim_tensor, etc.) is built on torch named tensors -- torch's
        # own docs mark this "an experimental feature... subject to change".
        # By torch 2.13, it looks entirely removed, not just changed: both
        # the `names=` tensor-factory kwarg (torch.ones(..., names=...)) AND
        # Tensor.rename() raise (TypeError / AttributeError respectively).
        # The exact version it broke in is unconfirmed. Bumping this needs a
        # real migration off named tensors as alan's plate-name<->tensor
        # bridge, not a one-line API swap.
        "torch>=2.0.0,<2.4",
        "numpy",
        "opt_einsum",
        "pytest",
    ],
    extras_requires=[
    ],
    include_package_data = True,
    platforms = "any",
)
