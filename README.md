# Equity and Autarky in PyPSA-Eur

Preprint: https://arxiv.org/abs/2007.08379

Social acceptance is a multifaceted consideration when planning future energy systems, yet often challenging to address endogeneously. One key aspect regards the spatial distribution of investments. Here, I evaluate the cost impact and changes in optimal system composition when development of infrastructure is more evenly shared among countries and regions in a fully renewable European power system. I deliberately deviate from the resource-induced cost optimum towards more equitable and self-sufficient solutions in terms of power generation. The analysis employs the open optimisation model PyPSA-Eur. I show that cost optimal solutions lead to very inhomogenous distributions of assets, but more uniform expansion plans can be achieved on a national level at little additional expense below 4%. Yet completely autarkic solutions, without power transmission, appear much more costly. 

## Usage

Install common `pypsa-eur` environment

```sh
cp config.pypsaeur.yaml pypsa-eur/config.yaml
conda activate pypsa-eur
cd equity-and-autarky/pypsa-eur
snakemake -j 99 solve_all_elec_networks
```