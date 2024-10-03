![image](https://github.com/user-attachments/assets/b127135b-f907-45c1-849b-70dd3ad9e612)

# Littlefield simulation tools
#### Scrape, save, load, and report data from Littlefield factory simulation, using Google Cloud [Functions](https://cloud.google.com/functions), [Scheduler](https://cloud.google.com/scheduler), [Storage](https://cloud.google.com/storage), and [BigQuery](https://cloud.google.com/bigquery).

## Options:
#### Save data as a CSV/Excel file to an existing GCS bucket
#### Load data to an existing BigQuery table
#### Automatically post snapshot reports to Discord

## Requirements:
- Existing Google Cloud Project
- Existing Google BigQuery dataset
- Existing Google Cloud Storage bucket
- Existing Discord Webhook

## To Run:

#### Part 1: Set Up Clound Function
Step 1: Create a Google Cloud Function (1st gen) with an HTTP trigger, with the following runtime environment variables:
- GROUP_ID: Group/team name
- GROUP_PW: Group/team password
- CLASS_URL: Littlefield login page url
- PROJECT_ID: Google Cloud Project ID
- DATASET_NAME: Google BigQuery Dataset name
- GCS_BUCKET: Google Cloud Storage bucket name
- DISCORD_WEBHOOK: Discord webhook url

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
  "csv": true,
  "excel": true,
  "bq_factory": true,
  "bq_standings": true,
  "bq_settings": true,
  "discord": true,
  "avg": 5
}
```

Step 5: Adjust the parameters based on desired usage

Step 6: Create the job

## Success!
...hopefully. Please let me know if you have any issues.







