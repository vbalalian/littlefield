![image](https://github.com/user-attachments/assets/b127135b-f907-45c1-849b-70dd3ad9e612)

# Littlefield simulation tools
#### Scrape, save, load, and report data from Littlefield factory simulation, using Google Cloud [Functions](https://cloud.google.com/functions), [Scheduler](https://cloud.google.com/scheduler), and [BigQuery](https://cloud.google.com/bigquery).

## Options:
#### Load data to an existing BigQuery table
#### Automatically post snapshot reports and useful charts to Discord

## Examples:
#### Daily Report:
![daily_report](https://github.com/user-attachments/assets/83725763-8daf-4999-ba91-2566ed7b88f5)
#### Demand Chart:
![demand_chart](https://github.com/user-attachments/assets/29d3dabf-af10-45db-b03a-83c383086e9b)
#### Utilization Chart:
![utilization_chart](https://github.com/user-attachments/assets/aa2d0e2c-c337-44c8-9b80-7be046499f2e)
#### Standings Chart:
![team_standings](https://github.com/user-attachments/assets/97d1193d-2857-42d2-853b-a3cfbd632b0d)

## Requirements:
- Existing Google Cloud Project
- Existing Google BigQuery dataset
- Existing Discord Webhook(s)

## To Run:

#### Part 1: Set Up Clound Function
Step 1: Create a Google Cloud Function (1st gen) with an HTTP trigger, with the following runtime environment variables:
- GROUP_ID: Group/team name
- GROUP_PW: Group/team password
- CLASS_URL: Littlefield login page url
- GCP_PROJECT_ID: Google Cloud Project ID
- BIGQUERY_DATASET_NAME: Google BigQuery Dataset name
- DISCORD_REPORT_WEBHOOK: Discord webhook url for quick report
- DISCORD_EXCEL_WEBHOOK: Discord webhook url for updated excel files
- DISCORD_DEMAND_UTIL_WEBHOOK: Discord webhook url for demand and utilization charts
- DISCORD_STANDINGS_WEBHOOK: Discord webhook url for standings chart

Step 2: Set runtime to **Python 3.11+** and entry point to **main**

Step 3: Copy [main.py](main.py) and [requirements.txt](requirements.txt) to function code

Step 4: Deploy the function

Step 5: Copy Trigger URL for use with Google Cloud Scheduler

#### Part 2: Set Up Cloud Scheduler
Step 1: Create a Google Cloud Scheduler job

Step 2: Set desired job run frequency using a [cron expression](https://crontab.guru/)

Step 3: Set the Target type to **HTTP** and copy the **Trigger URL** as the execution URL

Step 4: Copy the following into the body ('avg' = desired period count for discord report moving averages)
```
{
  "bigquery_factory": true,
  "bigquery_standings": true,
  "bigquery_settings": true,
  "discord_report": true,
  "discord_demand_chart": true,
  "discord_util_chart": true,
  "discord_standings_chart": true,
  "discord_excel": true,
  "avg": 5
}
```

Step 5: Adjust the parameters based on desired usage

Step 6: Create the job

## Success!
...hopefully. Please let me know if you have any issues.







