# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "great-tables==0.18.0",
#     "marimo",
#     "matplotlib==3.10.7",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import altair as alt
    import numpy as np
    from great_tables import GT, md, style, loc
    import matplotlib.pyplot as plt

    df = pd.read_csv("./Data.csv")
    return GT, alt, df, np, pd


@app.cell
def _(df):
    df.dropna(subset=["year", "Name", "gdp", "population", "temperature_change_from_co2", "co2_per_capita", "co2"],inplace=True)
    df2=df[["year", "Name", "gdp", "population", "co2_per_capita", "temperature_change_from_co2", "co2"]].query("year >= 1950")
    return (df2,)


@app.cell
def _(df2, np):
    final_df = df2.query("Name != 'World'")
    final_df.index = np.arange(1, len(final_df)+1)

    #final_df
    return (final_df,)


@app.cell
def _():
    #df2.info()
    return


@app.cell
def _():
    #print(df2['Name'].nunique())
    return


@app.cell
def _():
    #print(df2.shape)
    return


@app.cell
def _(alt, final_df):
    chart = alt.Chart(final_df)
    return


@app.cell
def _(final_df):
    by_emissions=final_df.sort_values(by="co2_per_capita", ascending=False).query("year == 2022")
    by_emissions
    return (by_emissions,)


@app.cell
def _(by_emissions):
    #by_emissions.iat[0,1]
    by_emissions.iat[0,4]
    return


@app.cell
def _():
    '''# Load the CO2 emissions data
    df_co2_pivot = pd.read_csv("Data.csv")
    # Apply filters
    # 1. Filter by year: 1950-2023
    df_filtered_pivot = df_co2_pivot[(df_co2_pivot["year"] >= 1950) & (df_co2_pivot["year"] <= 2023)].copy()
    # 2. Filter by Description: Only 'Country'
    df_filtered_pivot = df_filtered_pivot[df_filtered_pivot["Description"] == "Country"].copy()
    # 3. Filter by Name: Exclude strings ending with '(GCP)'
    df_filtered_pivot = df_filtered_pivot[~df_filtered_pivot["Name"].str.endswith("(GCP)", na=False)].copy()
    # 4. Remove rows with NaN values in our key columns
    df_filtered_pivot = df_filtered_pivot.dropna(subset=["Name", "co2", "population", "gdp"]).copy()

    # Create pivot table with Name as rows and co2, population, gdp as values
    # Using pivot_table instead of groupby
    pivot_table = pd.pivot_table(
        df_filtered_pivot,
        values=["co2", "population", "gdp"],
        index="Name",
        aggfunc="mean"
    ).round(2)

    # Sort by CO2 emissions (highest to lowest)
    pivot_table = pivot_table.sort_values("co2", ascending=False)

    # Get top 10 for display
    top_10_pivot = pivot_table.head(10)

    # Create display version with cleaned up labels
    formatted_pivot_table = top_10_pivot.copy()
    formatted_pivot_table.reset_index(inplace=True)
    formatted_pivot_table.columns = ["Name","Avg MtCO2", "Avg GDP", "Avg Population"]

    # Format for better display
    def format_number(x):
        if pd.isna(x):
            return "N/A"
        elif x >= 1e12:
            return f"${x/1e12:.1f}T"
        elif x >= 1e9:
            return f"${x/1e9:.1f}B"
        elif x >= 1e6:
            return f"${x/1e6:.1f}M"
        else:
            return f"${x:,.0f}"

    def format_population(x):
        if pd.isna(x):
            return "N/A"
        elif x >= 1e9:
            return f"{x/1e9:.2f}B"
        elif x >= 1e6:
            return f"{x/1e6:.1f}M"
        else:
            return f"{x:,.0f}"

    def format_co2(x):
        if pd.isna(x):
            return "N/A"
        else:
            return f"{x:,.1f}"


    # Apply formatting to each column
    formatted_pivot_table["Avg MtCO2"] = formatted_pivot_table["Avg MtCO2"].apply(format_co2)
    formatted_pivot_table["Avg GDP"] = formatted_pivot_table["Avg GDP"].apply(format_number)
    formatted_pivot_table["Avg Population"] = formatted_pivot_table["Avg Population"].apply(format_population)


    # Now formatted_pivot_table contains the formatted data as a DataFrame
    # You can display it with:
    #print(formatted_pivot_table)'''
    return


@app.cell
def _(GT, formatted_pivot_table):
    refined_table = GT(formatted_pivot_table)
    refined_table.tab_header(title="Top 10 CO2 Emitting Countries (1950-2022)").tab_options(
            heading_background_color="#3498db"
        ).opt_horizontal_padding(scale = 2).cols_align(align="center")
    return


@app.cell
def _(df2):
    avg_co2_by_country = df2.groupby("Name", as_index=False)["co2_per_capita"].mean()

    avg_co2_by_country = avg_co2_by_country[avg_co2_by_country["Name"] != "World"]

    avg_co2_by_country = avg_co2_by_country.sort_values(by="co2_per_capita", ascending=False)

    #print(avg_co2_by_country.head(10))
    print(avg_co2_by_country)
    return (avg_co2_by_country,)


@app.cell
def _(GT, avg_co2_by_country):
    avg_country_table = GT(avg_co2_by_country.head(15))
    avg_country_table.fmt_number(columns="co2_per_capita", compact=True).cols_label(co2_per_capita="CO2 Per Capita").tab_options(
            column_labels_background_color="#3498db"
        ).cols_align(columns="co2_per_capita", align="center" )
    return


@app.cell
def _():
    #world_df = df[["year", "Name", "temperature_change_from_co2"]].query("Name == 'World' and year >= 1950 ")
    #orld_df
    return


@app.cell(hide_code=True)
def _(np, world_df):
    xp=np.arange(min(world_df['year'])-1,max(world_df['year'])+50,1)
    x=world_df['year']
    y=world_df['temperature_change_from_co2']
    z2=np.polyfit(x,y,2)
    p2 = np.poly1d(z2)
    world_df.loc[len(world_df)]=[2070, "World", p2(2070)]
    #print(p2)
    world_df
    return


@app.cell
def _(world_df):
    world_df.drop(15)
    return


@app.cell
def _(alt, pd, world_df):
    source = pd.DataFrame({
        'x' : pd.to_datetime(world_df['year'], format='%Y'),
        'y' : (world_df["temperature_change_from_co2"])
    })

    base = alt.Chart(source).mark_line(color="#3498db", strokeWidth=4).encode(
        x=alt.X(
            "x:T",
            title="Year",
            axis=alt.Axis(format='%Y', tickCount='year')
        ),
        y=alt.Y('y', title= 'Temp (°C)'),
    )

    rect_data = pd.DataFrame([{'x1': pd.to_datetime(1950, format="%Y"), 'x2': pd.to_datetime(2070, format="%Y"), 'y1': 2, 'y2': 3}])
    highlight = alt.Chart(rect_data).mark_rect(
        color='red', 
        opacity=0.2
    ).encode(
        x='x1',
        x2='x2',
        y='y1',
        y2='y2'
    )

    world_chart = (base + highlight).properties(title="Global Temperature Change From CO2 vs Time").interactive()
    world_chart
    return


@app.cell
def _():
    #world_df.drop(index = 16, inplace = True)
    return


@app.cell
def _():
    import csv

    # Define regions with their countries
    regions_data = {
        'North America': [
            'United States', 'USA', 'United States of America', 'Canada', 'Mexico',
            'Greenland', 'Bermuda', 'Saint Pierre and Miquelon'
        ],

        'Central America': [
            'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 
            'Nicaragua', 'Panama', 'Bahamas', 'Cuba', 'Jamaica', 'Haiti', 
            'Dominican Republic', 'Puerto Rico', 'Trinidad and Tobago', 
            'Barbados', 'Saint Lucia', 'Grenada', 'Saint Vincent and the Grenadines',
            'Antigua and Barbuda', 'Dominica', 'Saint Kitts and Nevis',
            'Aruba', 'Curacao', 'Sint Maarten', 'Martinique', 'Guadeloupe',
            'Cayman Islands', 'Turks and Caicos', 'British Virgin Islands',
            'US Virgin Islands', 'Anguilla', 'Montserrat'
        ],

        'South America': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
            'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela',
            'French Guiana', 'Falkland Islands'
        ],

        'West Europe': [
        'Andorra', 'Austria', 'Belgium', 'Denmark', 'Finland', 'France', 
        'Germany', 'Greece', 'Iceland', 'Ireland', 'Italy', 'Liechtenstein', 
        'Luxembourg', 'Malta', 'Monaco', 'Netherlands', 'Norway', 'Portugal', 
        'San Marino', 'Spain', 'Sweden', 'Switzerland', 'United Kingdom', 'UK',
        'Great Britain', 'Vatican City', 'Faroe Islands', 'Gibraltar', 'Guernsey',
        'Isle of Man', 'Jersey', 'Svalbard and Jan Mayen'
        ],

        'East Europe': [
            'Albania', 'Belarus', 'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 
            'Cyprus', 'Czech Republic', 'Czechia', 'Estonia', 'Hungary', 'Kosovo', 
            'Latvia', 'Lithuania', 'Moldova', 'Montenegro', 'North Macedonia', 
            'Macedonia', 'Poland', 'Romania', 'Russia', 'Russian Federation', 
            'Serbia', 'Slovakia', 'Slovenia', 'Ukraine'
        ],

        'Middle East': [
            'Bahrain', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Kuwait', 'Lebanon',
            'Oman', 'Palestine', 'Qatar', 'Saudi Arabia', 'Syria', 'Turkey',
            'United Arab Emirates', 'UAE', 'Yemen', 'Armenia', 'Azerbaijan',
            'Georgia', 'Afghanistan', 'Pakistan'
        ],

        'North Africa': [
            'Algeria', 'Egypt', 'Libya', 'Morocco', 'Tunisia', 'Sudan',
            'Western Sahara', 'Mauritania'
        ],

        'West Africa': [
            'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', "Cote d'Ivoire",
            'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali',
            'Niger', 'Nigeria', 'Senegal', 'Sierra Leone', 'Togo'
        ],

        'East Africa': [
            'Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya',
            'Madagascar', 'Malawi', 'Mauritius', 'Mozambique', 'Rwanda',
            'Seychelles', 'Somalia', 'South Sudan', 'Tanzania', 'Uganda',
            'Zambia', 'Zimbabwe', 'Mayotte', 'Reunion'
        ],

        'South Africa': [
            'South Africa', 'Botswana', 'Lesotho', 'Namibia', 'Eswatini',
            'Swaziland', 'Angola'
        ],

        'East Asia': [
            'China', "People's Republic of China", 'Japan', 'South Korea',
            'North Korea', 'Mongolia', 'Taiwan', 'Hong Kong', 'Macau'
        ],

        'Southeast Asia': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Burma', 'Philippines', 'Singapore', 'Thailand', 'Timor-Leste',
            'East Timor', 'Vietnam'
        ],

        'Oceania': [
            'Australia', 'Fiji', 'Kiribati', 'Marshall Islands', 'Micronesia',
            'Nauru', 'New Zealand', 'Palau', 'Papua New Guinea', 'Samoa',
            'Solomon Islands', 'Tonga', 'Tuvalu', 'Vanuatu', 'Cook Islands',
            'French Polynesia', 'Guam', 'New Caledonia', 'Northern Mariana Islands',
            'American Samoa', 'Wallis and Futuna', 'Tokelau', 'Niue'
        ],

        'South Asia': [
            'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Sri Lanka'
        ]
    }

    # Create CSV file
    with open('country_regions.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['Name', 'Region'])

        # Write data
        for region, countries in regions_data.items():
            for country in countries:
                writer.writerow([country, region])

    print("CSV file 'country_regions.csv' has been created successfully!")
    print(f"Total countries: {sum(len(countries) for countries in regions_data.values())}")
    return


@app.cell
def _(final_df, pd):
    regions = pd.read_csv("./country_regions.csv")
    population_region_df = pd.merge(final_df, regions, how="inner", on="Name")
    return (population_region_df,)


@app.cell(hide_code=True)
def _(np, population_region_df):

    population_df = population_region_df.query("year > 2016").assign(log_population = lambda x: np.log(x['population']))
    return (population_df,)


@app.cell
def _(alt, population_df):
    region_list = population_df['Region'].unique().tolist()
    dropdown = alt.binding_select(
        options=region_list,
        name='Select Region: '
    )

    dropdown_selection = alt.selection_point(
        fields=['Region'],  
        bind=dropdown
    )

    click_selection = alt.selection_point(fields=['Region'])
    hover_selection = alt.selection_point(
        on="mouseover", fields=['Name'], empty=False
    )

    population_chart = (
        alt.Chart(population_df)
        .mark_circle()
        .encode(
            y=alt.Y('co2_per_capita', title="CO2 Emissions Per Capita (Tonnes Per Person)"),
            x=alt.X('log_population', title="log(Population)"),
            tooltip=['Name', 'co2_per_capita', 'population', 'year'], 
            color = alt.when(hover_selection)
                .then(alt.value("red"))
                .otherwise(
                    alt.Color("Region:N")
                ),
            size=alt.when(hover_selection)
                .then(alt.value(300))
                .when(click_selection)
                .then(alt.value(100))
                .otherwise(alt.value(50)),
            opacity=alt.when(hover_selection)
                .then(alt.value(1))
                .when(click_selection)
                .then(alt.value(1))
                .otherwise(alt.value(0.3))
        )
        .properties(
            width=900,
            height=500,
            title=alt.TitleParams(
                text="CO2 Emissions Per Capita vs Population (2017-2022)",
                fontSize=16,
                 )
        )
        .add_params(
        click_selection, hover_selection, dropdown_selection
        )
        .transform_filter(
            dropdown_selection
        )
        .interactive())
    population_chart
    return


@app.cell
def _(final_df):
    final_df
    return


@app.cell
def _(final_df, np):
    #GDP Coefficient
    x = final_df['year']
    y = final_df['gdp']

    correlation_matrix = np.corrcoef(x, y)
    print(correlation_matrix)
    return


@app.cell
def _(alt, pd):
    df_co2 = pd.read_csv("Data.csv")
    df_filtered = df_co2[(df_co2["year"] >= 1950) & (df_co2["year"] <= 2023)].copy()
    df_filtered = df_filtered[df_filtered["Description"] == "Country"].copy()
    df_filtered = df_filtered[~df_filtered["Name"].str.endswith("(GCP)", na=False)].copy()
    df_filtered = df_filtered.dropna(subset=["Name", "co2", "gdp"]).copy()

    # Calculate global averages by year (mean across all countries)
    yearly_averages = (
        df_filtered.groupby("year").agg({"co2": "mean", "gdp": "mean"}).reset_index()
    )

    # Define tick values for every 5 years from 1950-2022
    tick_values = [1950, 1955, 1960, 1965, 1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022]

    # CO2 chart
    co2_chart = (
        alt.Chart(yearly_averages)
        .mark_line(
            color="#e74c3c", strokeWidth=3
        )
        .encode(
            x=alt.X(
                "year:O",
                title="Year",
                axis=alt.Axis(
                    labelAngle=-45, values=tick_values, labelExpr="datum.value + ''"
                ),
            ),
            y=alt.Y(
                "co2:Q",
                title="Average CO2 Emissions per Country (MtCO2)",
                axis=alt.Axis(titleColor="#e74c3c", labelColor="#e74c3c"),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("co2:Q", title="Avg CO2 per Country (MtCO2)", format=",.1f"),
                alt.Tooltip("gdp:Q", title="Avg GDP per Country (USD)", format="$,.0f"),
            ],
        )
    )

    # GDP chart (independent y-axis)
    gdp_chart = (
        alt.Chart(yearly_averages)
        .mark_line(
            color="#3498db",
            strokeWidth=3
        )
        .encode(
            x=alt.X(
                "year:O", axis=alt.Axis(values=tick_values, labelExpr="datum.value + ''")
            ),
            y=alt.Y(
                "gdp:Q",
                title="Average GDP per Country (USD)",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(titleColor="#3498db", labelColor="#3498db", format="$.0f", labelExpr="'$' + datum.value/1000000000 + 'B'"),
            ),
            tooltip=[
                alt.Tooltip("year:O", title="Year"),
                alt.Tooltip("co2:Q", title="Avg CO2 per Country (MtCO2)", format=",.1f"),
                alt.Tooltip("gdp:Q", title="Avg GDP per Country (USD)", format="$,.0f"),
            ],
        )
    )

    # Combine with independent y-axes
    gdp_co2_chart = (
        alt.layer(co2_chart, gdp_chart)
        .resolve_scale(y="independent")
        .properties(
            width=900,
            height=450,
            title=alt.TitleParams(
                text="Average GDP vs CO2 per Country (1950-2022)",
                fontSize=16,
            ),
        )
    )

    gdp_co2_chart
    return (yearly_averages,)


@app.cell
def _(yearly_averages):
    yearly_averages
    return


@app.cell
def _(alt, df2, pd):
    world_df = df2[df2["Name"] == "World"]

    source = pd.DataFrame({
        'x': pd.to_datetime(world_df['year'], format="%Y"),
        'y': world_df['co2_per_capita']
    })

    #baseline co2 emissions
    co2_line = (
        alt.Chart(source)
        .mark_line(color="#3498db", strokeWidth = 3)
        .encode(
            x=alt.X('x:T', title="Year", axis=alt.Axis(format='%Y')),
            y=alt.Y('y:Q', title="CO₂ Emissions (million tonnes)", axis=alt.Axis(titleColor="#3498db")),
            tooltip=[alt.Tooltip('x:T', title='Year', format='%Y'), alt.Tooltip('y:Q', title='CO₂ per capita')]
        )
    )

    source = pd.DataFrame({
        'x': pd.to_datetime(world_df['year'], format="%Y"),
        'y': world_df['temperature_change_from_co2']
    })
    #line for temp change with second axis
    temp_line = (
        alt.Chart(source)
        .mark_line(color="#e74c3c", strokeWidth=3)
        .encode(
            x=alt.X('x:T', axis=alt.Axis(format='%Y')),
            y=alt.Y(
                "y:Q",
                title="Temperature Change from CO₂ (°C)",
                axis=alt.Axis(titleColor="#e74c3c")
            ),
            tooltip=[alt.Tooltip('x:T', title='Year', format='%Y'), alt.Tooltip('y:Q', title='Temperature Change')]
        )
    )


    #combine both lines, layered graph with two y-axes
    co2_vs_temp = alt.layer(co2_line, temp_line).resolve_scale(y='independent').properties(
        title="CO₂ Emissions vs. Temperature Change (World)",
        width=900,
        height=450
    )

    co2_vs_temp
    return (world_df,)


@app.cell
def _(np, world_df):
    # Example data
    variable1 = world_df['co2_per_capita']
    variable2 = world_df['temperature_change_from_co2']

    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(variable1, variable2)

    # The correlation coefficient between variable1 and variable2 is at index [0, 1] or [1, 0]
    correlation_coefficient = correlation_matrix[0, 1]

    print(f"Correlation coefficient: {correlation_coefficient}")
    return


@app.cell
def _(np, world_df):
    list = np.array(world_df['temperature_change_from_co2'])
    print(((list[-1] - list[0])/list[0])*100)
    return


@app.cell
def _(np, world_df):
    list2 = np.array(world_df['co2_per_capita'])
    print(((list2[-1] - list2[0])/list2[0])*100)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
