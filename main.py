import os
import mechanize
from google.cloud import bigquery
from bs4 import BeautifulSoup
import http.cookiejar as cookielib
import pandas as pd
import numpy as np
import dataframe_image as dfi
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

def scrape_data(browser:mechanize.Browser) -> pd.DataFrame:
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

def scrape_standings(browser:mechanize.Browser) -> list[dict]:
    standings_url = 'https://op.responsive.net/Littlefield/Standing'
    status_url = 'http://op.responsive.net/Littlefield/LTStatus'
    standings_soup = BeautifulSoup(browser.open(standings_url), 'lxml')
    status_soup = BeautifulSoup(browser.open(status_url), 'lxml')
    table = standings_soup.find('table', {'id': 'standingTable'})
    rows = table.find('tbody').find_all('tr')
    team_standings = []
    day = status_soup.find_all('b')[0].next_sibling.strip()

    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()
        team_name = cols[1].text.strip()
        cash_balance = cols[2].text.strip()
        team_standings.append({
            'DAY': int(day),
            'RANK': int(rank), 
            'TEAM': str(team_name), 
            'CASH_BALANCE': float(cash_balance.replace(',',''))
        })
    return team_standings

def get_team_info(standings:list[dict], team_name:str=group_id):
    for team in standings:
        if team['TEAM'].lower() == team_name.lower():
            return team
    return None

def csv_to_bucket(df:pd.DataFrame, bucket:str, filename:str='data.csv'):
    '''Saves DataFrame to csv in GCS bucket'''
    print(f'Saving CSV file {filename} to bucket {bucket}...')
    df.to_csv(f'gs://{bucket}/{filename}')
    print('...complete.')

def excel_to_bucket(df:pd.DataFrame, bucket:str, filename:str='data.xlsx'):
    '''Saves Dataframe to excel in GCS bucket'''
    print(f'Writing {filename} to bucket {bucket}...')
    df.to_excel(f'gs://{bucket}/{filename}')
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
        f'Avg ({avg}d) Job Demand':[df.loc[present_day-avg:present_day, 'JOBIN'].mean()],
        'Station 1 Util (%)':[df.loc[present_day, 'S1UTIL']*100],
        f'Avg ({avg}d) St 1 Util (%)':[df.loc[present_day-avg:present_day, 'S1UTIL'].mean()*100],
        'Station 2 Util (%)':[df.loc[present_day, 'S2UTIL']*100],
        f'Avg ({avg}d) St 2 Util (%)':[df.loc[present_day-avg:present_day, 'S2UTIL'].mean()*100],
        'Station 3 Util (%)':[df.loc[present_day, 'S3UTIL']*100],
        f'Avg ({avg}d) St 3 Util (%)':[df.loc[present_day-avg:present_day, 'S3UTIL'].mean()*100],
        }, index=[team_name]).transpose()
    report_df = report_df.style.format(precision=0)
    return report_df

def post_report_to_discord(df:pd.DataFrame, discord_webhook:str, filename:str='chart.png', font_size:int=10) -> requests.Response:
    '''Posts image to Discord webhook, returns requests.Response'''
    print('Posting report image to Discord...')
    temp_file = '/tmp/' + filename
    dfi.export(df, filename=temp_file, table_conversion='matplotlib', fontsize=font_size)
    with open(temp_file, 'rb') as image:
        response = requests.post(discord_webhook,
                files={filename: image})
    print('...complete.')
    return response

def main(request):
    # Connect to Littlefield
    browser = Browser()
    login(class_url, group_id, group_pw, browser)

    # Scrape Littlefield data
    factory_df = scrape_data(browser)
    standings = scrape_standings(browser)
    rank = get_team_info(standings, group_id)['RANK']
    standings_df = pd.DataFrame(standings)

    # Parse request parameters
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload', 400

    # Load data based on request parameters
    actions = {
        'csv': lambda: csv_to_bucket(factory_df, gcs_bucket),
        'excel': lambda: excel_to_bucket(factory_df, gcs_bucket),
        'bq_factory': lambda: load_to_bigquery(
            factory_df, project_id, dataset_name, factory_table
        ),
        'bq_standings': lambda: load_to_bigquery(
            standings_df, project_id, dataset_name, standings_table
        ),
        'discord': lambda: post_report_to_discord(
            daily_report(factory_df, group_id, rank, request_json.get('avg')), webhook
            )
    }
    for key, action in actions.items():
        if request_json.get(key):
            action()

    return 'OK', 200

if __name__ == '__main__':
    main()