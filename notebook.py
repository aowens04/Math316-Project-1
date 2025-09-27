# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "pandas==2.3.2",
# ]
# ///

import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt

    df = pd.read_csv("./Data.csv")
    return alt, df


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell
def _(df):
    df.dropna(subset=["year", "Name", "gdp", "population", "temperature_change_from_co2", "co2_including_luc"],inplace=True)
    df2=df[["year", "Name", "gdp", "population", "co2_including_luc", "temperature_change_from_co2"]].query("year > 1950")
    df2.sort_values(by="temperature_change_from_co2", ascending=False)
    return (df2,)


@app.cell
def _(df2):
    df2.info()
    return


@app.cell
def _(alt, df2):
    chart = alt.Chart(df2)
    return (chart,)


@app.cell
def _(chart):
    chart.mark_point().encode(
        x='co2_including_luc',
        y='gdp'
    )
    return


@app.cell
def _(chart):
    chart.mark_point().encode(
        x='co2_including_luc',
        y='population'
    )
    return


@app.cell
def _(chart):
    chart.mark_point().encode(
        x='co2_including_luc',
        y='temperature_change_from_co2'
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
