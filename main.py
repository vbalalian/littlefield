import os
import mechanize
from google.cloud import bigquery
from bs4 import BeautifulSoup
import http.cookiejar as cookielib
import pandas as pd
import numpy as np
import dataframe_image as dfi
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import requests
import json

# Littlefield website info
class_url = os.getenv('CLASS_URL')
group_id = os.getenv('GROUP_ID')
group_pw = os.getenv('GROUP_PW')

# Google Cloud project info
project_id = os.getenv('PROJECT_ID')
dataset_name = os.getenv('DATASET_NAME')
gcs_bucket = os.getenv('GCS_BUCKET')
factory_table = 'raw_factory'
standings_table = 'raw_standings'
settings_table = 'raw_settings'

# Discord webhook
webhook = os.getenv('DISCORD_WEBHOOK')

def Browser() -> mechanize.Browser:
    '''Returns mechanize.Browser object with set cookiejar'''
    cj = cookielib.CookieJar()
    br = mechanize.Browser()
    br.set_cookiejar(cj)
    return br

def login(url:str, id:str, pw:str, browser:mechanize.Browser):
    '''Logs into Littlefield simulation URL'''
    print('Logging in...')
    browser.open(url)
    browser.select_form(nr=0)
    browser.form['id'] = id
    browser.form['password'] = pw
    browser.submit()

def scrape_current_day(browser:mechanize.Browser) -> int:
    status_url = 'http://op.responsive.net/Littlefield/LTStatus'
    status_soup = BeautifulSoup(browser.open(status_url), 'lxml')
    day = status_soup.find_all('b')[0].next_sibling.strip()
    return int(day)

def scrape_full_data(browser:mechanize.Browser) -> pd.DataFrame:
    '''Scrapes data from Littlefield categorical URLS, returns a DataFrame'''
    # Set variables
    simple_categories = ["INV", "AVGINV", "CASH","JOBIN", "JOBREJECTS", "JOBQ", "JOBWIP", "S1Q","S2Q","S3Q","S1UTIL","S2UTIL","S3UTIL"]
    contract_categories = ["JOBT","JOBREV","JOBOUT"]
    dataset = {}

    # Scrape simple_categories
    for cat in simple_categories:
        print(f'Scraping {cat}')
        url = "http://op.responsive.net/Littlefield/Plot?data=%s&x=all" % cat
        soup = BeautifulSoup(browser.open(url), "lxml")
        data = soup.find_all("script")[6].string
        data = data.split("\n")[4].split("'")[3].split()
        values = []
        for i in range(len(data)):
            if i % 2 == 0 and "." not in data[i]:
                values.append(data[i+1])
        dataset[cat] = values

    # Scrape contract_categories
    for cat in contract_categories:
        print(f'Scraping {cat}')
        url = "http://op.responsive.net/Littlefield/Plot?data=%s&x=all" % cat
        soup = BeautifulSoup(browser.open(url), "lxml")
        data = soup.find_all("script")[6].string
        for i in range(3):
            try:
                data = data.split("\n")[i+4].split("'")[5].split()
                values = []
                for j in range(len(data)):
                    if j % 2 == 0 and "." not in data[j]:
                        values.append(data[j+1])
                dataset[cat+str(i+1)] = values
            except:
                # Fill unused Contract 2/3 columns with NaNs
                values = [np.nan for _ in range(len(dataset['INV']))]
                dataset[cat+str(i+1)] = values
                continue

    df = pd.DataFrame(dataset, dtype=float)
    df['DAY'] = df.index+1
    # Move 'DAY' to front
    df = df[ ['DAY'] + [ col for col in df.columns if col != 'DAY' ] ]
    df.index += 1
    return df

def scrape_current_standings(browser:mechanize.Browser, current_day:int) -> pd.DataFrame: ################### ADD TO MAIN
    standings_url = 'https://op.responsive.net/Littlefield/Standing'
    standings_soup = BeautifulSoup(browser.open(standings_url), 'lxml')
    table = standings_soup.find('table', {'id': 'standingTable'})
    rows = table.find('tbody').find_all('tr')
    team_standings = []
    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()
        team_name = cols[1].text.strip()
        cash_balance = cols[2].text.strip()
        team_standings.append({
            'DAY': current_day,
            'RANK': int(rank), 
            'TEAM': str(team_name), 
            'CASH_BALANCE': float(cash_balance.replace(',',''))
        })

    standings_df = pd.DataFrame(team_standings)    
    return standings_df

def scrape_daily_settings(browser:mechanize.Browser, current_day:int) -> pd.DataFrame:
    dataset = {'DAY': current_day}

    # Scrape station settings
    for station_no in range(1, 4):
        url = 'https://op.responsive.net/Littlefield/StationMenu?id=%s' % station_no
        soup = BeautifulSoup(browser.open(url), 'lxml')
        values = []
        for row in soup.find_all('b')[1:5]:
            values.append(row.next_sibling.strip())
        dataset[f'S{station_no}_machines'] = int(values[0])
        dataset[f'S{station_no}_sch_policy'] = values[1]
        dataset[f'S{station_no}_pur_price'] = float(values[2].replace('$', '').replace(',', '').strip())
        dataset[f'S{station_no}_ret_price'] = float(values[3].replace('$', '').replace(',', '').strip())

    # Scrape orders menu
    url = 'https://op.responsive.net/Littlefield/OrdersMenu'
    soup = BeautifulSoup(browser.open(url), 'lxml')
    values = []
    for row in soup.find_all('b')[1:5]:
        values.append(row.next_sibling.split()[0].strip())
    for row in soup.find_all('dd')[:3]:
        values.append(row.contents[0].split()[3])
    dataset['max_wip'] = int(values[0])
    dataset['kits_per_job'] = int(values[1])
    dataset['lot_size'] = int(values[2])
    dataset['current_contract'] = int(values[3])
    dataset['quoted_lead_time'] = float(values[4])
    dataset['max_lead_time'] = float(values[5])
    dataset['rev_per_order'] = float(values[6])
    
    # Scrape materials menu
    url = 'https://op.responsive.net/Littlefield/MaterialMenu'
    soup = BeautifulSoup(browser.open(url), 'lxml')
    values = []
    for row in soup.find_all('b')[1:6]:
        values.append(row.next_sibling.replace('$','').replace(',','').split()[0].strip())
    dataset['unit_cost'] = float(values[0])
    dataset['order_cost'] = float(values[1])
    dataset['lead_time'] = float(values[2])
    dataset['reorder_point'] = float(values[3])
    dataset['order_quantity'] = float(values[4])

    return pd.DataFrame(dataset, index=[current_day])

def scrape_team_info(standings:pd.DataFrame, team_name:str=group_id):
    teams_list = standings['TEAM'].str.lower().values
    if team_name.lower() in teams_list:
        return standings[standings['TEAM'].str.lower() == team_name.lower()].to_dict('records')[0]
    return {}

def csv_to_bucket(df:pd.DataFrame, bucket:str, filename:str='data.csv'):
    '''Saves DataFrame to csv in GCS bucket'''
    print(f'Saving CSV file {filename} to bucket {bucket}...')
    df.to_csv(f'gs://{bucket}/{filename}')
    print('...complete.')

def excel_to_bucket(df:pd.DataFrame, bucket:str, filename:str='data.xlsx'):
    '''Saves Dataframe to excel in GCS bucket'''
    print(f'Writing {filename} to bucket {bucket}...')
    df.to_excel(f'gs://{bucket}/{filename}', index=False)
    print('...complete.')

def load_to_bigquery(df:pd.DataFrame, project:str, dataset:str, table:str):
    '''Loads DataFrame into BigQuery table'''
    table_id = f'{project}.{dataset}.{table}'
    print(f'Loading to BigQuery table:{table_id}')
    client = bigquery.Client()
    try:
        query = f'SELECT * FROM {table_id};'
        job = client.query(query)
        result = job.result()
        existing_records = {row.DAY for row in result}
        new_records_df = df[df['DAY'].apply(lambda x: x not in existing_records)]
        job_config = bigquery.LoadJobConfig(write_disposition = "WRITE_APPEND")
        job = client.load_table_from_dataframe(new_records_df, table_id, job_config=job_config)
        result = job.result()
    except:
        job_config = bigquery.LoadJobConfig(write_disposition = "WRITE_TRUNCATE")
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        result = job.result()
    print(f'Loaded {job.output_rows} rows into {table_id}.')

def daily_report(df:pd.DataFrame, team_name:str, rank:int|None=None, average:int|None=None) -> pd.DataFrame:
    '''Transforms DataFrame into eye friendly report based on values dict'''
    present_day = df['DAY'].max()
    avg = average if average else 5
    rank = rank if rank else 'na'
    report_df = pd.DataFrame({
        'Day':[present_day],
        'Rank': [rank], 
        'Cash':[df.loc[present_day, 'CASH']],
        'Inventory':[df.loc[present_day, 'INV']],
        'Job Demand':[df.loc[present_day, 'JOBIN']],
        f'^ {avg}d avg':[df.loc[present_day-avg:present_day, 'JOBIN'].mean()],
        'Station 1 Util':f"{round(df.loc[present_day, 'S1UTIL']*100)} %",
        f' ^ {avg}d avg':f"{round(df.loc[present_day-avg:present_day, 'S1UTIL'].mean()*100)} %",
        'Station 2 Util':f"{round(df.loc[present_day, 'S2UTIL']*100)} %",
        f'  ^ {avg}d avg':f"{round(df.loc[present_day-avg:present_day, 'S2UTIL'].mean()*100)} %",
        'Station 3 Util':f"{round(df.loc[present_day, 'S3UTIL']*100)} %",
        f'   ^ {avg}d avg':f"{round(df.loc[present_day-avg:present_day, 'S3UTIL'].mean()*100)} %",
        }, index=[team_name]).transpose()
    report_df = report_df.style.format(precision=0)
    return report_df

def post_report_to_discord(df:pd.DataFrame, discord_webhook:str, current_day:int|None=None, font_size:int=10) -> requests.Response:
    '''Posts report image to Discord webhook, returns requests.Response'''
    print('Posting report image to Discord...')
    filename = f'day{current_day}-report' if current_day else 'report'
    temp_file = '/tmp/' + filename + '.png'
    dfi.export(df, filename=temp_file, table_conversion='matplotlib', fontsize=font_size)
    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response

def post_demand_chart_to_discord(df:pd.DataFrame, discord_webhook:str, current_day:int|None=None) -> requests.Response:
    '''Posts job demand chart to Discord webhook, returns requests.Response'''
    print('Posting job demand chart to Discord...')
    filename = f'day{current_day}-demand-chart' if current_day else 'demand-chart'
    temp_file = '/tmp/' + filename + '.png'

    # Plot jobs in
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.grid(True)
    plt.plot(df['DAY'], df['JOBIN'], linewidth=2)

    # Add trendline with standard deviation
    z = np.polyfit(df['DAY'], df['JOBIN'], 1)
    p = np.poly1d(z)
    x = range(1, df['DAY'].max()+25)
    std_dev = np.std(df['JOBIN'])
    plt.plot(x, p(x), color="red", linestyle="--")
    plt.fill_between(x, p(x) - std_dev, p(x) + std_dev, alpha=0.3)

    # Add station capacities
    df['JOBOUT'] = df['JOBOUT1'].combine_first(df['JOBOUT2']).combine_first(df['JOBOUT3'])
    avg_jobs_out = df['JOBOUT'].mean()
    colors = ['black', 'gray', 'silver']
    for i in range(1, 4):
        avg_util = df[f'S{i}UTIL'].mean()
        capacity = avg_jobs_out/avg_util
        plt.axhline(y=capacity, color=colors[i-1], linestyle='--', label=f'S{i}CAPACITY')
        plt.text(0, capacity+.1, f'S{i}CAPACITY', fontsize=6)

    plt.xlim(0, df['DAY'].max()+25)
    plt.xlabel('Day')
    plt.ylabel('Jobs In')
    plt.title("Job Demand")
    plt.savefig(temp_file)
    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response 
    
def post_util_chart_to_discord(df:pd.DataFrame, discord_webhook:str, current_day:int|None=None) -> requests.Response:
    '''Posts station utilization charts to Discord webhook, returns requests.Response'''
    print('Posting station utilization chart to Discord...')
    filename = f'util-chart-day{current_day}' if current_day else 'util-chart'
    temp_file = '/tmp/' + filename + '.png'
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.grid(True)
    colors = ['green', 'purple', 'orange']
    for i in range (1, 4):
        plt.plot(df['DAY'], df[f'S{i}UTIL']*100, color=colors[i-1], linewidth=2, label=f'S{i}UTIL')
        z = np.polyfit(df['DAY'], df[f'S{i}UTIL']*100, 1)
        p = np.poly1d(z)
        x = range(1, df['DAY'].max()+25)
        plt.plot(x, p(x), color=colors[i-1], linestyle="--")
    plt.xlim(0, df['DAY'].max()+25)
    plt.ylim(-5, 105)
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('Utilization %')
    plt.title(f"Station Utilization")
    plt.savefig(temp_file)
    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response 

def post_standings_chart_to_discord(discord_webhook:str, project:str, dataset:str, table:str, current_day:int|None=None) -> requests.Response:
    '''Posts standings chart to Discord webhook, returns requests.Response'''
    print('Posting standings chart to Discord...')
    filename = f'day{current_day}-standings-chart' if current_day else 'standings-chart'
    temp_file = '/tmp/' + filename + '.png'

    client = bigquery.Client()

    # Query to extract the latest two days of standings and cash balance
    query = f"""
    SELECT DAY, RANK, TEAM, CASH_BALANCE
    FROM `{project}.{dataset}.{table}`
    WHERE DAY IN (SELECT MAX(DAY) FROM `{project}.{dataset}.{table}`)
    OR DAY IN (SELECT MAX(DAY) - 1 FROM `{project}.{dataset}.{table}`)
    ORDER BY TEAM, DAY
    """

    # Execute the query
    query_job = client.query(query)
    df = query_job.to_dataframe()

    # Pivot the data to compare ranks between two days
    pivot_df = df.pivot(index='TEAM', columns='DAY', values='RANK').reset_index()

    # Handle missing data (if only one day's data is available)
    if len(pivot_df.columns) == 2:
        pivot_df.columns = ['TEAM', 'RANK_TODAY']
        pivot_df['RANK_YESTERDAY'] = pivot_df['RANK_TODAY']
    else:
        pivot_df.columns = ['TEAM', 'RANK_YESTERDAY', 'RANK_TODAY']

    # Calculate rank changes
    def rank_change(row):
        if row['RANK_TODAY'] < row['RANK_YESTERDAY']:
            return f"+{row['RANK_YESTERDAY'] - row['RANK_TODAY']}"
        elif row['RANK_TODAY'] > row['RANK_YESTERDAY']:
            return f"-{row['RANK_TODAY'] - row['RANK_YESTERDAY']}"
        else:
            return "-"

    pivot_df['RANK_CHANGE'] = pivot_df.apply(rank_change, axis=1)

    # Merge with cash balance data
    cash_df = df[df['DAY'] == df['DAY'].max()][['TEAM', 'RANK', 'CASH_BALANCE']]
    result_df = pivot_df.merge(cash_df, on='TEAM')

    # Generate colors using a colormap
    cmap = plt.get_cmap('tab20', len(result_df))  # Get a colormap with distinct colors
    colors = cmap(range(len(result_df)))  # Assign a color to each team

    # Create horizontal bar chart with cash on x-axis and rank on y-axis
    plt.figure(figsize=(10, 8))
    plt.barh(result_df['TEAM'] + ' ' + result_df['RANK'].astype(str) + '(' + result_df['RANK_CHANGE'] + ')', result_df['CASH_BALANCE'], color=colors)
    plt.xlabel('Cash Balance (Millions)')
    plt.ylabel('Team')
    plt.title(f'Team Standings - Day {df["DAY"].max()}')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.axvline(x=result_df['CASH_BALANCE'].max(), color='gold', linestyle='--')
    plt.text(result_df['CASH_BALANCE'].max(), 26, f"${result_df['CASH_BALANCE'].max():,.0f}", fontsize=14, rotation=270, va='center')
    plt.gca().invert_yaxis() 

    # Set the x-axis limits
    plt.xlim(1000000, max(df['CASH_BALANCE'].max()+15000, 2000000))

    # Format x-ticks in millions
    formatter = mticker.FuncFormatter(lambda x, _: f'{round(float(x/1000000), 2)}M')
    plt.gca().xaxis.set_major_formatter(formatter)

    plt.savefig(temp_file)

    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response 

def post_excel_to_discord(df:pd.DataFrame, discord_webhook:str, current_day:int|None=None) -> requests.Response:
    '''Posts excel file to Discord webhook, returns requests.Response'''
    print('Posting excel file to Discord...')
    filename = f'littlefield-day{current_day}' if current_day else 'littlefield'
    temp_file = '/tmp/' + filename + '.xlsx'
    df.to_excel(temp_file, index=False)
    with open(temp_file, 'rb') as file:
        response = requests.post(discord_webhook,
                files={filename: file})
    print('...complete.')
    return response

def main(request):
    # Connect to Littlefield
    browser = Browser()
    login(class_url, group_id, group_pw, browser)

    # Scrape Littlefield data
    current_day = scrape_current_day(browser)
    factory_df = scrape_full_data(browser)
    standings_df = scrape_current_standings(browser, current_day)
    daily_settings_df = scrape_daily_settings(browser, current_day)
    team_info = scrape_team_info(standings_df, group_id)
    rank = team_info['RANK']

    # Parse request parameters
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload', 400

    # Load data based on request parameters
    actions = {
        'csv_to_bucket': lambda: csv_to_bucket(factory_df, gcs_bucket),
        'excel_to_bucket': lambda: excel_to_bucket(factory_df, gcs_bucket),
        'bigquery_factory': lambda: load_to_bigquery(
            factory_df, project_id, dataset_name, factory_table
        ),
        'bigquery_standings': lambda: load_to_bigquery(
            standings_df, project_id, dataset_name, standings_table
        ),
        'bigquery_settings': lambda: load_to_bigquery(
            daily_settings_df, project_id, dataset_name, settings_table
        ),
        'discord_demand_chart': lambda: post_demand_chart_to_discord(
            df=factory_df, current_day=current_day, discord_webhook=webhook
        ),
        'discord_util_chart': lambda: post_util_chart_to_discord(
            df=factory_df, current_day=current_day, discord_webhook=webhook
        ),
        'discord_standings_chart': lambda: post_standings_chart_to_discord(
            discord_webhook=webhook, project=project_id, dataset=dataset_name, 
            table=standings_table, current_day=current_day
        ),
        'discord_report': lambda: post_report_to_discord(
            daily_report(df=factory_df, team_name=group_id, rank=rank, average=request_json.get('avg')), 
            current_day=current_day, discord_webhook=webhook
        ),
        'discord_excel': lambda: post_excel_to_discord(factory_df, webhook, current_day)
    }
    for key, action in actions.items():
        if request_json.get(key):
            action()

    return 'OK', 200

if __name__ == '__main__':
    main()