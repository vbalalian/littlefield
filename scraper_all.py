import os
import mechanize
from google.cloud import bigquery
from bs4 import BeautifulSoup
import http.cookiejar as cookielib
import pandas as pd
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
bigquery_table = os.getenv('TABLE_NAME')
gcs_bucket = os.getenv('GCS_BUCKET')

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
    url_list = ["INV", "AVGINV", "CASH","JOBIN", "JOBREJECTS", "JOBQ", "JOBWIP", "S1Q","S2Q","S3Q","S1UTIL","S2UTIL","S3UTIL"]
    url_list_4col = ["JOBT","JOBREV","JOBOUT"]
    LF_DATA = {}

    # Scrape url_list
    for url in url_list:
        print(f'Scraping {url}')
        lf_url = "http://op.responsive.net/Littlefield/Plot?data=%s&x=all" % url
        soup = BeautifulSoup(browser.open(lf_url), "lxml")
        data = soup.find_all("script")[6].string
        data = data.split("\n")[4].split("'")[3].split()
        values = []
        for i in range(len(data)):
            if i % 2 == 0 and "." not in data[i]:
                values.append(data[i+1])
        LF_DATA[url] = values

    # Scrape url_list_4col
    for url in url_list_4col:
        print(f'Scraping {url}')
        lf_url = "http://op.responsive.net/Littlefield/Plot?data=%s&x=all" % url
        soup = BeautifulSoup(browser.open(lf_url), "lxml")
        data = soup.find_all("script")[6].string
        for i in range(3):
            try:
                data = data.split("\n")[i+4].split("'")[5].split()
                values = []
                for j in range(len(data)):
                    if j % 2 == 0 and "." not in data[j]:
                        values.append(data[j+1])
                LF_DATA[url+str(i+1)] = values
            except:
                # Fill Contract 2 and 3 columns with 0s
                values = [0 for _ in range(len(LF_DATA['INV']))]
                LF_DATA[url+str(i+1)] = values
                continue

    df = pd.DataFrame(LF_DATA, dtype=float)
    df.index += 1
    df.index.name = 'DAY'
    return df

def scrape_standings(browser:mechanize.Browser) -> list[dict]:
    '''Scrapes Littlefield team standings, returns a list of dicts'''
    standings_url = 'https://op.responsive.net/Littlefield/Standing'
    soup = BeautifulSoup(browser.open(standings_url), 'lxml')
    table = soup.find('table', {'id': 'standingTable'})
    rows = table.find('tbody').find_all('tr')
    team_standings = []
    for row in rows:
        cols = row.find_all('td')
        rank = cols[0].text.strip()
        team_name = cols[1].text.strip()
        cash_balance = cols[2].text.strip()
        team_standings.append({
            'rank': rank, 
            'team': team_name, 
            'cash_balance': cash_balance
        })
    return team_standings

def get_team_info(standings:list[dict], team_name:str=group_id):
    '''Returns rank, team name, and cash balance of specified team from standings'''
    for team in standings:
        if team['team'].lower() == team_name.lower():
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
    table_id = f'{project}.{dataset}.{table}'
    print(f'Loading to BigQuery table:{table_id}')
    client = bigquery.Client()
    job_config = bigquery.LoadJobConfig(
        write_disposition = "WRITE_TRUNCATE"
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    print('...complete.')
    return job.result()

def daily_report(df:pd.DataFrame, team_name:str, rank:int|None=None, average:int|None=None) -> pd.DataFrame:
    '''Transforms DataFrame into eye friendly report based on values dict'''
    present_day = df.index.max()
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
    data = scrape_data(browser)
    standings = scrape_standings(browser)
    rank = get_team_info(standings, group_id)['rank']

    # Parse request parameters
    request_json = request.get_json(silent=True)
    if not request_json:
        return 'Invalid JSON payload', 400

    # Load data based on request parameters
    actions = {
        'csv': lambda: csv_to_bucket(data, gcs_bucket),
        'excel': lambda: excel_to_bucket(data, gcs_bucket),
        'bigquery': lambda: load_to_bigquery(
            data, project_id, dataset_name, bigquery_table
            ),
        'discord': lambda: post_report_to_discord(
            daily_report(data, group_id, rank, request_json.get('avg')), webhook
            )
    }
    for key, action in actions.items():
        if request_json.get(key):
            action()

    return 'OK', 200

if __name__ == '__main__':
    main()