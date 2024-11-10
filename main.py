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
    
    # Combine contract category columns
    for category in contract_categories:
        df[category] = df[f'{category}1'].combine_first(df[f'{category}2']).combine_first(df[f'{category}3'])

    df['DAY'] = df.index+1
    # Move 'DAY' to front
    df = df[ ['DAY'] + [ col for col in df.columns if col != 'DAY' ] ]
    df.index += 1
    return df

def scrape_current_standings(browser:mechanize.Browser, current_day:int) -> pd.DataFrame:
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

def get_settings_historical_data(project:str, dataset:str, table:str):

    client = bigquery.Client()

    # Query to extract the latest two days of standings and cash balance
    query = f"""
    SELECT DAY, S1_machines, S2_machines, S3_machines
    FROM `{project}.{dataset}.{table}`
    ORDER BY DAY
    """

    # Execute the query
    query_job = client.query(query)
    df = query_job.to_dataframe()

    return df

def daily_report(
        factory_df:pd.DataFrame, 
        team_name:str, 
        project:str,
        dataset:str,
        table:str, 
        rank:str|None=None, 
        average:int|None=None) -> pd.DataFrame:
    '''Transforms DataFrame into eye friendly report based on values dict'''
    present_day = factory_df['DAY'].max()
    avg = average if average else 5
    rank = rank if rank else 'na'

    # Add station capacities
    avg_jobs_out = factory_df['JOBOUT'].mean()
    machines_1 = pd.DataFrame({ # machine counts for days before game start
        'DAY': [i for i in range(1, 50)],
        'S1_machines': [1 for _ in range(49)], 
        'S2_machines': [1 for _ in range(49)], 
        'S3_machines': [1 for _ in range(49)]
        })
    
    machines_2 = get_settings_historical_data(project=project, dataset=dataset, table=table)
    machines_all = pd.concat([machines_1, machines_2], ignore_index=True)
    current_machines = machines_all.loc[machines_all['DAY'] == machines_all['DAY'].max()]
    capacities, machines = [], []
    for i in range(1, 4):
        avg_util = factory_df[f'S{i}UTIL'].mean()
        avg_capacity = avg_jobs_out/avg_util
        avg_machine_count = machines_all[f'S{i}_machines'].mean()
        machine_count = current_machines[f'S{i}_machines'].values[0]
        machines.append(machine_count)
        cap_per_machine = avg_capacity/avg_machine_count
        capacity = cap_per_machine*machine_count
        capacities.append(round(capacity, 2))

    report_df = pd.DataFrame({
        'Day':[present_day],
        'Rank': [rank], 
        'Cash':f"${factory_df.loc[present_day, 'CASH']*1000:,.0f}",
        'Inventory':f"{factory_df.loc[present_day, 'INV']:,.0f}",
        f'Demand ({avg}d MA)':f"{round([factory_df.loc[present_day, 'JOBIN']][0])} ({round([factory_df.loc[present_day-avg:present_day, 'JOBIN'].mean()][0], 2)})",
        f'S1 Util ({avg}d MA)':f"{round(factory_df.loc[present_day, 'S1UTIL']*100)}% ({round(factory_df.loc[present_day-avg:present_day, 'S1UTIL'].mean()*100)}%)",
        f'S2 Util ({avg}d MA)':f"{round(factory_df.loc[present_day, 'S2UTIL']*100)}% ({round(factory_df.loc[present_day-avg:present_day, 'S2UTIL'].mean()*100)}%)",
        f'S3 Util ({avg}d MA)':f"{round(factory_df.loc[present_day, 'S3UTIL']*100)}% ({round(factory_df.loc[present_day-avg:present_day, 'S3UTIL'].mean()*100)}%)",
        'Capacities': f"{capacities[0]}, {capacities[1]}, {capacities[2]}",
        'Machines': f"{machines[0]}, {machines[1]}, {machines[2]}",
        'Jobs Out': f"{round(factory_df.loc[present_day, 'JOBOUT'])}",
        'Avg Lead Time': f"{factory_df.loc[present_day, 'JOBT']}",
        'Revenue': f"${factory_df.loc[present_day, 'JOBREV']*factory_df.loc[present_day, 'JOBOUT']:,.0f}",
        'Profit': f"${(factory_df.loc[present_day, 'JOBREV']*factory_df.loc[present_day, 'JOBOUT'])-(factory_df.loc[present_day, 'JOBOUT']*600):,.0f}"
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

def post_demand_chart_to_discord(factory_df:pd.DataFrame, 
                                 settings_df:pd.DataFrame, 
                                 discord_webhook:str,
                                 project:str,
                                 dataset:str,
                                 table:str
                                 ) -> requests.Response:
    '''Posts job demand chart to Discord webhook, returns requests.Response'''

    print('Posting job demand chart to Discord...')
    current_day = factory_df['DAY'].max()
    filename = f'day{current_day}-demand-chart' if current_day else 'demand-chart'
    temp_file = '/tmp/' + filename + '.png'

    # Plot jobs in
    plt.figure(figsize=(6, 4), layout='constrained')
    plt.grid(True)
    plt.plot(factory_df['DAY'], factory_df['JOBIN'], linewidth=2)

    # Add trendline with standard deviation
    stage_1 = factory_df[factory_df['DAY'] <= 170]
    z = np.polyfit(stage_1['DAY'], stage_1['JOBIN'], 1)
    p = np.poly1d(z)
    std_dev = np.std(stage_1['JOBIN'])
    ft_end = current_day + 14

    # Trendline before day 170
    x1 = range(1, min(ft_end, 170) + 1)
    plt.plot(x1, p(x1), color="red", linestyle="--")
    plt.fill_between(x1, p(x1) - std_dev, p(x1) + std_dev, color="C0", alpha=0.3)

    # Trendline after day 170
    if ft_end >= 170:
        if current_day <= 177:
            x2 = range(170, ft_end+1)
            plt.plot(x2, [p(170)] * len(x2), color="red", linestyle="--")
            plt.fill_between(x2, [p(170) - std_dev] * len(x2), [p(170) + std_dev] * len(x2), color="C0", alpha=0.3)
        elif current_day > 177:
            stage_2 = factory_df[factory_df['DAY'] > 170]
            z2 = np.polyfit(stage_2['DAY'], stage_2['JOBIN'], 1)
            p2 = np.poly1d(z2)
            x2 = range(170, min(ft_end, 268))
            plt.plot(x2, p2(x2), color="red", linestyle="--")
            plt.fill_between(x2, p2(x2) - std_dev, p2(x2) + std_dev, color="C0", alpha=0.3)

    # Add station capacities
    avg_jobs_out = factory_df['JOBOUT'].mean()
    machines_1 = pd.DataFrame({ # machine counts for days before game start
        'DAY': [i for i in range(1, 50)],
        'S1_machines': [1 for _ in range(49)], 
        'S2_machines': [1 for _ in range(49)], 
        'S3_machines': [1 for _ in range(49)]
        })
    machines_2 = get_settings_historical_data(project=project, dataset=dataset, table=table)
    machines_all = pd.concat([machines_1, machines_2], ignore_index=True)
    current_machines = machines_all[machines_all['DAY'] == machines_all['DAY'].max()]
    colors = ['black', 'gray', 'silver']
    for i in range(1, 4):
        avg_util = factory_df[f'S{i}UTIL'].mean()
        avg_capacity = avg_jobs_out/avg_util
        avg_machine_count = machines_all[f'S{i}_machines'].mean()
        machine_count = current_machines[f'S{i}_machines'].values[0]
        cap_per_machine = avg_capacity/avg_machine_count
        capacity = cap_per_machine*machine_count
        plt.axhline(y=capacity, color=colors[i-1], linestyle='--', label=f'S{i}CAPACITY')
        plt.text((i*12)-12, capacity+.1, f'S{i}CAPACITY', fontsize=6)

    # plt.xlim(0, factory_df['DAY'].max()+25)
    plt.xlabel('Day')
    plt.ylabel('Jobs In')
    plt.title("Job Demand")
    plt.savefig(temp_file)
    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response 
    
def post_util_chart_to_discord(factory_df:pd.DataFrame, discord_webhook:str, current_day:int|None=None, average:int|None=None) -> requests.Response:
    '''Posts station utilization charts to Discord webhook, returns requests.Response'''
    print('Posting station utilization chart to Discord...')
    filename = f'util-chart-day{current_day}' if current_day else 'util-chart'
    temp_file = '/tmp/' + filename + '.png'
    plt.figure(figsize=(6, 4), layout='constrained')
    plt.grid(True)
    colors = ['green', 'purple', 'orange']
    ma = average if average else 5
    for i in range (1, 4):
        plt.plot(factory_df['DAY'], factory_df[f'S{i}UTIL']*100, color=colors[i-1], linewidth=3, label=f'S{i}UTIL', alpha=.40)
        plt.plot(factory_df['DAY'], factory_df[f'S{i}UTIL'].rolling(ma).mean()*100, color=colors[i-1], linestyle="--", label=f'S{i} {ma}p MA')

    # plt.xlim(0, factory_df['DAY'].max()+25)
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
    WITH RankedTeams AS (
        SELECT 
            DAY, 
            TEAM, 
            CASH_BALANCE,
            RANK() OVER (PARTITION BY DAY ORDER BY CASH_BALANCE DESC) AS RANK
        FROM `{project}.{dataset}.{table}`
        WHERE DAY IN (SELECT MAX(DAY) FROM `{project}.{dataset}.{table}`)
        OR DAY IN (SELECT MAX(DAY) - 1 FROM `{project}.{dataset}.{table}`)
    )

    SELECT * FROM RankedTeams
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
        try:
            if row['RANK_TODAY'] < row['RANK_YESTERDAY']:
                return f"+{row['RANK_YESTERDAY'] - row['RANK_TODAY']}"
            elif row['RANK_TODAY'] > row['RANK_YESTERDAY']:
                return f"-{row['RANK_TODAY'] - row['RANK_YESTERDAY']}"
            else:
                return "-"
        except:
            return "-"

    pivot_df['RANK_CHANGE'] = pivot_df.apply(rank_change, axis=1)

    # Merge with cash balance data
    cash_df = df[df['DAY'] == df['DAY'].max()][['TEAM', 'RANK', 'CASH_BALANCE']]

    # Identify tied ranks and add a "T-" prefix to the rank where necessary
    cash_df['TIED_RANK'] = cash_df.duplicated(subset='RANK', keep=False)  # Check for ties
    cash_df['DISPLAY_RANK'] = cash_df.apply(lambda row: f"T-{row['RANK']}" if row['TIED_RANK'] else str(row['RANK']), axis=1)

    # Merge the cash data back into pivot_df
    result_df = pivot_df.merge(cash_df[['TEAM', 'RANK', 'CASH_BALANCE', 'DISPLAY_RANK']], on='TEAM')

    # Generate colors using a colormap
    cmap = plt.get_cmap('tab20', len(result_df))
    colors = cmap(range(len(result_df))) 

    # Create horizontal bar chart with cash on x-axis and rank on y-axis
    plt.figure(figsize=(10, 8))
    plt.barh(result_df['TEAM'] + ' ' + result_df['DISPLAY_RANK'] + '(' + result_df['RANK_CHANGE'] + ')', result_df['CASH_BALANCE'], color=colors)
    plt.xlabel('Cash Balance (Millions)')
    plt.ylabel('Team')
    plt.title(f'Team Standings - Day {df["DAY"].max()}')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.axvline(x=result_df['CASH_BALANCE'].max(), color='gold', linestyle='--')
    plt.text(result_df['CASH_BALANCE'].max(), 26, f"${result_df['CASH_BALANCE'].max():,.0f}", fontsize=14, rotation=270, va='center')
    plt.gca().invert_yaxis() 

    # Set the x-axis limits
    plt.xlim(0, max(df['CASH_BALANCE'].max()+15000, 2000000))

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

    # Parse request parameters
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload', 400

    # Discord webhook
    if request_json.get('test'):
        report_webhook = os.getenv('TEST_DISCORD_REPORT_WEBHOOK')
        excel_webhook = os.getenv('TEST_DISCORD_EXCEL_WEBHOOK')
        demand_util_webhook = os.getenv('TEST_DISCORD_DEMAND_UTIL_WEBHOOK')
        standings_webhook = os.getenv('TEST_DISCORD_STANDINGS_WEBHOOK')
    else:
        report_webhook = os.getenv('DISCORD_REPORT_WEBHOOK')
        excel_webhook = os.getenv('DISCORD_EXCEL_WEBHOOK')
        demand_util_webhook = os.getenv('DISCORD_DEMAND_UTIL_WEBHOOK')
        standings_webhook = os.getenv('DISCORD_STANDINGS_WEBHOOK')

    # Scrape Littlefield data
    current_day = scrape_current_day(browser)
    factory_df = scrape_full_data(browser)
    standings_df = scrape_current_standings(browser, current_day)
    daily_settings_df = scrape_daily_settings(browser, current_day)
    team_info = scrape_team_info(standings_df, group_id)
    rank = f"T-{team_info['RANK']}" if standings_df[standings_df['RANK'] == team_info['RANK']].shape[0] > 1 else f"{team_info['RANK']}"

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
        'discord_report': lambda: post_report_to_discord(
            daily_report(factory_df=factory_df, team_name=group_id, rank=rank, project=project_id, 
                dataset=dataset_name, table=settings_table, average=request_json.get('avg')), 
            current_day=current_day, discord_webhook=report_webhook
        ),
        'discord_demand_chart': lambda: post_demand_chart_to_discord(
            factory_df=factory_df, settings_df=daily_settings_df, discord_webhook=demand_util_webhook, 
            project=project_id, dataset=dataset_name, table=settings_table
        ),
        'discord_util_chart': lambda: post_util_chart_to_discord(
            factory_df=factory_df, current_day=current_day, discord_webhook=demand_util_webhook, average=request_json.get('avg')
        ),
        'discord_standings_chart': lambda: post_standings_chart_to_discord(
            discord_webhook=standings_webhook, project=project_id, dataset=dataset_name, 
            table=standings_table, current_day=current_day
        ),
        'discord_excel': lambda: post_excel_to_discord(factory_df, excel_webhook, current_day)
    }
    for key, action in actions.items():
        if request_json.get(key):
            action()

    return 'OK', 200

if __name__ == '__main__':
    main()