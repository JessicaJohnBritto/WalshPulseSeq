
# Walsh Pulse Sequence [WPS] Protocol 

This is developed as part of my 2024 summer internship at LPMMC, UGA-CNRS, Grenoble, France in Prof. Benoît Vermersch's research group.

WPSProtocol _(WPS == Walsh Pulse Sequence)_ makes use of Walsh Functions to generate Pulse Sequences for selective interactions among qubits based on the underlying XY Hamiltonian.

WPSProtocol contains codes for implementing the protocol described in this paper _Votto, M., Zeiher, J., & Vermersch, B. (2023). Robust universal quantum processors in spin systems via Walsh pulse sequences. arXiv preprint arXiv:2311.10600._

## Theory
The theory behind this protocol can be found at [WalshPulseSeqProtocol:A python package for Dynamical Decoupling of Spin Systems using Walsh Functions](https://drive.google.com/file/d/1KWIZfiM3YbbgYqnDeojgH0q4sAX7M3V1/view?usp=sharing).

## Installation

Create a virtual environment

```bash
conda create -n <virtual_env name>
```

Installing the package
```bash
python -m pip install WPSProtocol
```

## Run Locally

Make sure to create a virtual environment. To create a conda environment, using the following command.
```bash
  conda create -n <virtual_env name>
``` 
Clone the project
- Using https

```bash
  git clone https://github.com/JessicaJohnBritto/WalshPulseSeq.git
```
- Using ssh

```bash
  git clone git@github.com:JessicaJohnBritto/WalshPulseSeq.git
```

## Contributing

Follow the steps given in `Run Locally`. 

By following the command below, you will be in the development mode.

_Note: Be inside the directory where pyproject.toml is before running the following command. For eg - here `WalshPulseSeq` is the directory, therefore, the path variable should be `../WalshPulseSeq`_.
```bash
  python -m pip install -e .
```

To install the package using git, use the following command.

```bash
  python -m pip install git+https://github.com/JessicaJohnBritto/WalshPulseSeq.git#egg=WPSProtocol

```
