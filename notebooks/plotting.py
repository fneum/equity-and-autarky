#!/usr/bin/env python
# coding: utf-8

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import yaml
plt.style.use("bmh")

sys.path.insert(0, os.getcwd()+"/../losses/scripts")
sys.path.insert(0, os.getcwd()+"/../loss-approximation/scripts")
import plotting.collection as clt
from plotting.utils import aggregate_costs, assign_carriers

nodes = sys.argv[1]
add = sys.argv[2]

with open("config.pypsaeur.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

d = f"graphics/{nodes}/"
if not os.path.exists(d):
    os.makedirs(d)

def plot_lorentz(ns, keys=[], by='bus', fn=None):

    fig, ax = plt.subplots(figsize=(4,4))
    
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="k", linestyle=":")

    for k in keys:
        n = ns[k]
        cumshare = cumulative_share(n, by=by)
        plt.plot(cumshare.load, cumshare.energy, label=f"{100*k}%")

    plt.xlabel("Cumulative Share of\n Electricity Demand")
    plt.ylabel("Cumulative Share of\n Electricity Generation")
    plt.legend(title="Share of Consumption\nProduced", frameon=False)

    if fn is not None:
        plt.savefig(fn, bbox_inches="tight")


def cumulative_share(n, by="bus"):

    n.loads["load"] = n.loads_t.p.multiply(n.snapshot_weightings, axis=0).sum()
    n.generators["energy"] = n.generators_t.p.multiply(
        n.snapshot_weightings, axis=0
    ).sum()

    if by == "country":
        n.loads["country"] = n.loads.apply(lambda x: x.bus[:2], axis=1)
        n.generators["country"] = n.generators.apply(lambda x: x.bus[:2], axis=1)

    energy = n.generators.groupby(by).energy.sum()
    load = n.loads.groupby(by).load.sum()

    df = pd.concat([(energy / energy.sum()), (load / load.sum())], axis=1)
    df.sort_values(by="energy", inplace=True)

    return df.cumsum()


def plot_area(costs, colors, add, fn=None):
    fig, ax = plt.subplots(figsize=(5,4))
    costs.T.plot.area(color=colors, ax=ax)
    plt.legend(ncol=1, bbox_to_anchor=(1.05, 1))
    plt.xlim([0,100])
    plt.yticks(np.arange(0,281,25));
    plt.ylim([0,280])
    plt.ylabel("Total System Costs [bn Euro p.a.]")
    lbl = "National" if add == "c" else "Nodal"
    plt.xlabel(f"Share of {lbl} Consumption Produced [%]")
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')


def calculate_imbalance(n):
    gen = n.generators_t.p.multiply(n.snapshot_weightings, axis=0).groupby(n.generators.bus.map(n.buses.country), axis=1).sum().sum() / 1e6
    load = n.loads_t.p_set.multiply(n.snapshot_weightings, axis=0).groupby(n.loads.bus.map(n.buses.country), axis=1).sum().sum() / 1e6
    inflow = n.storage_units_t.inflow.groupby(n.storage_units.bus.map(n.buses.country), axis=1).sum().sum() / 1e6
    inflow = inflow.reindex(gen.index).fillna(0.)
    imbalance = pd.DataFrame({
        "generation": gen + inflow,
        "consumption": load
    })
    imbalance.index.name = 'country'
    return imbalance


def plot_imbalance(imb, fn=None):
    fig, ax = plt.subplots(figsize=(6,2))
    imb.eval("100 * generation / consumption").sort_values().plot.bar()
    plt.axhline(100, c="k", linestyle=":")
    plt.ylabel("share of national\nconsumption\nproduced [%]")
    #ax.text(23, 230, "DK = 1400%")
    plt.ylim([0,250])
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

def plot_cost_by_country(n, relative=True, fn=None):
    
    load = (n.snapshot_weightings @ n.loads_t.p_set).groupby(n.buses.country).sum()

    cost = cost_by_country(n)
    if relative:
        df = cost.divide(load, axis=0)
        label = "Relative investment\n[Euro/MWh]"
        ylim = [0,200]
    else:
        df = cost / 1e9
        label = "Total investment\n[bn Euro p.a.]"
        ylim = [0,50]
    
    fig, ax = plt.subplots(figsize=(6,2))

    df = df.reindex(df.sum(axis=1).sort_values().index, axis=0)

    df.plot.bar(stacked=True, ax=ax, color=["midnightblue", "seagreen", "gold"])
    
    avg = df.sum(axis=1).mean()
    plt.axhline(avg, c="k", linestyle=":", label=f"average: {avg:.1f}")
    plt.ylabel(label)
    plt.ylim(ylim)
    plt.legend()
    
    if fn is not None:
        plt.savefig(fn, bbox_inches='tight')

def cost_by_country(n):
    cts = n.buses.country.unique()

    g = n.generators.eval("p_nom_opt * capital_cost").groupby(n.generators.bus.map(n.buses.country)).sum()

    s = n.storage_units.eval("p_nom_opt * capital_cost").groupby(n.storage_units.bus.map(n.buses.country)).sum()

    ln0 = n.lines.eval("0.5 * s_nom_opt * capital_cost").groupby(n.lines.bus0.map(n.buses.country)).sum().reindex(cts).fillna(0)

    ln1 = n.lines.eval("0.5 * s_nom_opt * capital_cost").groupby(n.lines.bus1.map(n.buses.country)).sum().reindex(cts).fillna(0)

    lk0 = n.links.eval("0.5 * p_nom_opt * capital_cost").groupby(n.links.bus0.map(n.buses.country)).sum().reindex(cts).fillna(0)

    lk1 = n.links.eval("0.5 * p_nom_opt * capital_cost").groupby(n.links.bus1.map(n.buses.country)).sum().reindex(cts).fillna(0)

    ln = ln0 + ln1
    lk = lk0 + lk1

    return pd.DataFrame({
        "generation": g,
        "transmission": ln + lk,
        "storage": s
    })

def calculate_costs_and_revenues(n):
    lmp = n.buses_t.marginal_price
    cost = n.loads_t.p_set.multiply(n.snapshot_weightings, axis=0).multiply(lmp)

    genlmp = lmp.reindex(columns=n.generators.bus)
    genlmp.columns = n.generators.index
    genrevenue = n.generators_t.p.multiply(n.snapshot_weightings, axis=0).multiply(genlmp)

    stolmp = lmp.reindex(columns=n.storage_units.bus)
    stolmp.columns = n.storage_units.index
    stocost = - n.storage_units_t.p.where(n.storage_units_t.p<0).fillna(0.).multiply(n.snapshot_weightings, axis=0).multiply(stolmp)
    storevenue = n.storage_units_t.p.where(n.storage_units_t.p>0).fillna(0.).multiply(n.snapshot_weightings, axis=0).multiply(stolmp)

    lklmp0 = lmp.reindex(columns=n.links.bus0)
    lklmp0.columns = n.links.index
    lklmp1 = lmp.reindex(columns=n.links.bus1)
    lklmp1.columns = n.links.index

    lkcr = n.links_t.p0.multiply(n.snapshot_weightings, axis=0).multiply(lklmp1-lklmp0)

    lnlmp0 = lmp.reindex(columns=n.lines.bus0)
    lnlmp0.columns = n.lines.index
    lnlmp1 = lmp.reindex(columns=n.lines.bus1)
    lnlmp1.columns = n.lines.index

    lncr = n.lines_t.p0.multiply(n.snapshot_weightings, axis=0).multiply(lnlmp1-lnlmp0)
    
    return cost, genrevenue, stocost, storevenue, lkcr, lncr


def calculate_welfare(cost, genrevenue, stocost, storevenue, lkcr, lncr):
    return pd.Series({
        "load cost": cost.sum().sum() / 1e9,
        "generator revenue": - genrevenue.sum().sum() / 1e9,
        "storage cost": + stocost.sum().sum() / 1e9,
        "storage revenue": - storevenue.sum().sum() / 1e9,
        "line congestion rent": - lncr.sum().sum() / 1e9,
        "link congestion rent": - lkcr.sum().sum() / 1e9
    })


def calculate_country_welfare(cost, genrevenue, stocost, storevenue, lkcr, lncr):
    return pd.DataFrame({
        "load_cost": cost.groupby(n.buses.country, axis=1).sum().sum() / 1e9,
        "generator revenue": - genrevenue.groupby(n.generators.bus.map(n.buses.country), axis=1).sum().sum() / 1e9,
        "storage cost": + stocost.groupby(n.storage_units.bus.map(n.buses.country), axis=1).sum().sum() / 1e9,
        "storage revenue": - storevenue.groupby(n.storage_units.bus.map(n.buses.country), axis=1).sum().sum() / 1e9,
    })


imb2018 = pd.read_csv("data/imbalance.csv", index_col=0).reindex(config["countries"]).dropna()
plot_imbalance(imb2018, fn=f"{d}imbalance-2018.pdf")


step = 0.1
csvargs = {"float_format": "%.2f"}

keys = [np.round(i,1) for i in np.arange(step,1.1,step)]
ns = {i: pypsa.Network(f"pypsa-eur/results/networks/elec_s_{nodes}_ec_lcopt_2H-EQ{i}{add}.nc") for i in keys}
ns[0.0] = pypsa.Network(f"pypsa-eur/results/networks/elec_s_{nodes}_ec_lcopt_2H.nc")

for key, n in ns.items():
    assign_carriers(n)

for key, n in ns.items():
    clt.plot_network(n, fn=f"{d}map-{key}{add}.pdf")

plot_lorentz(ns, keys=[0.0, 0.4, 0.6, 0.8], by='bus', fn=f"{d}lorentz-bus-{add}.pdf")

plot_lorentz(ns, keys=[0.0, 0.4, 0.6, 0.8], by="country", fn=f"{d}lorentz-country-{add}.pdf")

costs = pd.DataFrame(
    {k: aggregate_costs(n) for k, n in ns.items()}
).sort_index(axis=1).drop("load")

costs.to_csv(f"{d}costs-{add}.csv", **csvargs)

colors = costs.index.map(config["plotting"]["tech_colors"])

costs = costs.rename(index=config["plotting"]["nice_names"])
costs.columns = [100*i for i in costs.columns]

plot_area(costs, colors, add, fn=f"{d}sensitivity-{key}{add}.pdf")

for key, n in ns.items():
    imb = calculate_imbalance(n)
    plot_imbalance(imb, fn=f"{d}imbalance-{key}-{add}.pdf")

for key, n in ns.items():
    plot_cost_by_country(n, relative=False, fn=f"{d}countrycost-abs-{key}-{add}.pdf")
    plot_cost_by_country(n, relative=True, fn=f"{d}countrycost-rel-{key}-{add}.pdf")

welfare = pd.concat({k: calculate_welfare(*calculate_costs_and_revenues(n)) for k, n in ns.items()}, axis=1).T.sort_index()
welfare.to_csv(f"{d}welfare.csv",**csvargs)

for k, n in ns.items():
    cw = calculate_country_welfare(*calculate_costs_and_revenues(n))
    cw.to_csv(f"{d}welfare-by-country-{k}.csv",**csvargs)

